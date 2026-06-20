import logging
from pathlib import Path
from claim_extractor import extract_claim
from image_analyzer import analyze_image
from evidence_engine import get_evidence_requirements_text
from risk_engine import build_risk_flags
from decision_engine import generate_claim_status

def process_claim(client, row):
    user_id = row["user_id"]
    image_paths_str = row["image_paths"]
    user_claim = row["user_claim"]
    claim_object = row["claim_object"]

    # 1. Split image paths
    image_paths = [p.strip() for p in image_paths_str.split(";") if p.strip()]
    num_images = len(image_paths)

    # 2. Extract claimed issue and part
    claim = extract_claim(client, user_claim, claim_object)
    claimed_issue = claim["claimed_issue"]
    claimed_part = claim["claimed_part"]
    
    logging.info(f"Processing claim for user {user_id}. Object: {claim_object}. Claimed: {claimed_issue} on {claimed_part}")

    # 3. Get evidence requirements
    evidence_requirements = get_evidence_requirements_text(
        claim_object, claimed_issue, claimed_part, num_images
    )

    # 4. Analyze images
    vision = analyze_image(
        client, image_paths, claim_object, claimed_issue, claimed_part, evidence_requirements
    )

    # 5. Build risk flags
    risk_flags_list = build_risk_flags(user_id, vision)
    risk_flags_str = ";".join(risk_flags_list)

    # 6. Generate final status and justification
    status, justification = generate_claim_status(
        claimed_issue, claimed_part, vision, risk_flags_list
    )

    # 7. Format supporting image IDs
    sids = [str(x).strip() for x in vision.get("supporting_image_ids", []) if x]
    original_stems = [Path(p).stem for p in image_paths]
    valid_sids = [s for s in sids if s in original_stems]
    
    # Fallback to the first image ID if supported but list is empty
    if not valid_sids and status == "supported" and original_stems:
        valid_sids = [original_stems[0]]
        
    supporting_image_ids_str = ";".join(valid_sids) if valid_sids else "none"

    # Convert booleans to "true" / "false" string values
    evidence_standard_met_str = "true" if vision.get("evidence_standard_met", True) else "false"
    valid_image_str = "true" if vision.get("valid_image", True) else "false"

    return {
        "user_id": user_id,
        "image_paths": image_paths_str,
        "user_claim": user_claim,
        "claim_object": claim_object,
        "evidence_standard_met": evidence_standard_met_str,
        "evidence_standard_met_reason": vision.get("evidence_standard_met_reason", "").strip(),
        "risk_flags": risk_flags_str,
        "issue_type": vision.get("issue_type", "unknown"),
        "object_part": vision.get("object_part", "unknown"),
        "claim_status": status,
        "claim_status_justification": justification,
        "supporting_image_ids": supporting_image_ids_str,
        "valid_image": valid_image_str,
        "severity": vision.get("severity", "unknown")
    }