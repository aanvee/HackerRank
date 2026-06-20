import logging
from PIL import Image
from pathlib import Path
from prompts import VISION_ANALYSIS_PROMPT
from utils import (
    parse_json_safely, retry_api_call, ALLOWED_PARTS, ALLOWED_ISSUES, 
    get_repo_root, API_USAGE, get_cache_key, get_cached_response, set_cached_response
)

@retry_api_call(max_retries=3, initial_backoff=2)
def analyze_image(client, image_paths, claim_object, claimed_issue, claimed_part, evidence_requirements):
    # Check cache first
    cache_key = get_cache_key("vision", str(image_paths), claim_object, claimed_issue, claimed_part, evidence_requirements)
    cached = get_cached_response(cache_key)
    if cached:
        logging.info(f"Using cached vision analysis for image paths: '{image_paths}'")
        return cached

    if isinstance(image_paths, str):
        paths_list = [p.strip() for p in image_paths.split(";") if p.strip()]
    else:
        paths_list = image_paths

    repo_root = get_repo_root()
    contents = []

    parts = ALLOWED_PARTS.get(claim_object, ["unknown"])
    allowed_parts_str = ", ".join(parts)

    formatted_prompt = VISION_ANALYSIS_PROMPT.format(
        claim_object=claim_object,
        claimed_issue=claimed_issue,
        claimed_part=claimed_part,
        evidence_requirements=evidence_requirements,
        allowed_parts=allowed_parts_str
    )

    contents.append(formatted_prompt)

    images_loaded = 0
    for path_str in paths_list:
        p = Path(path_str)
        if p.is_absolute():
            abs_path = p
        elif path_str.startswith("dataset"):
            abs_path = repo_root / path_str
        else:
            abs_path = repo_root / "dataset" / path_str

        if abs_path.exists():
            img_id = abs_path.stem  # e.g. "img_1"
            try:
                img = Image.open(abs_path)
                contents.append(f"Image ID: {img_id}")
                contents.append(img)
                images_loaded += 1
            except Exception as e:
                logging.error(f"Failed to open image at {abs_path}: {e}")
        else:
            logging.warning(f"Image file does not exist at {abs_path}")

    if images_loaded == 0:
        logging.error("No images could be loaded for vision analysis.")
        return get_fallback_vision_result()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents
    )

    # Increment usage metrics
    API_USAGE["calls"] += 1
    API_USAGE["images"] += images_loaded
    if response.usage_metadata:
        API_USAGE["prompt_tokens"] += response.usage_metadata.prompt_token_count
        API_USAGE["completion_tokens"] += response.usage_metadata.candidates_token_count

    text = response.text.strip()
    try:
        data = parse_json_safely(text)
        
        issue_type = str(data.get("issue_type", "unknown")).lower().strip()
        object_part = str(data.get("object_part", "unknown")).lower().strip()
        severity = str(data.get("severity", "unknown")).lower().strip()
        
        valid_image = data.get("valid_image", True)
        if isinstance(valid_image, str):
            valid_image = valid_image.lower() == "true"
            
        evidence_standard_met = data.get("evidence_standard_met", True)
        if isinstance(evidence_standard_met, str):
            evidence_standard_met = evidence_standard_met.lower() == "true"
            
        risk_flags = [str(flag).lower().strip() for flag in data.get("risk_flags", [])]
        supporting_image_ids = [str(sid).strip() for sid in data.get("supporting_image_ids", [])]

        if issue_type not in ALLOWED_ISSUES:
            issue_type = "unknown"
        if object_part not in parts:
            object_part = "unknown"
        if severity not in ["none", "low", "medium", "high", "unknown"]:
            severity = "unknown"
            
        result = {
            "issue_type": issue_type,
            "object_part": object_part,
            "severity": severity,
            "valid_image": valid_image,
            "risk_flags": risk_flags,
            "evidence_standard_met": evidence_standard_met,
            "evidence_standard_met_reason": data.get("evidence_standard_met_reason", ""),
            "supporting_image_ids": supporting_image_ids,
            "claim_status_suggestion": data.get("claim_status_suggestion", "not_enough_information"),
            "claim_status_justification": data.get("claim_status_justification", "")
        }
        
        # Save to cache
        set_cached_response(cache_key, result)
        return result
    except Exception as e:
        logging.error(f"Failed to parse vision model response. Response text: '{text}'. Error: {e}")
        return get_fallback_vision_result()

def get_fallback_vision_result():
    return {
        "issue_type": "unknown",
        "object_part": "unknown",
        "severity": "unknown",
        "valid_image": False,
        "risk_flags": ["manual_review_required"],
        "evidence_standard_met": False,
        "evidence_standard_met_reason": "VLM JSON parsing or model generation error.",
        "supporting_image_ids": [],
        "claim_status_suggestion": "not_enough_information",
        "claim_status_justification": "Failed to extract structured data from vision analysis."
    }