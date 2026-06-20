import logging
from google import genai
from prompts import CLAIM_EXTRACTION_PROMPT
from utils import (
    parse_json_safely, retry_api_call, ALLOWED_PARTS, ALLOWED_ISSUES, 
    API_USAGE, get_cache_key, get_cached_response, set_cached_response
)

@retry_api_call(max_retries=3, initial_backoff=2)
def extract_claim(client, conversation, claim_object):
    # Check cache first
    cache_key = get_cache_key("extract", conversation, claim_object)
    cached = get_cached_response(cache_key)
    if cached:
        logging.info(f"Using cached claim extraction for conversation prefix: '{conversation[:30]}...'")
        return cached

    parts = ALLOWED_PARTS.get(claim_object, ["unknown"])
    allowed_parts_str = ", ".join(parts)
    allowed_issues_str = ", ".join(ALLOWED_ISSUES)
    
    prompt = CLAIM_EXTRACTION_PROMPT.format(
        claim_object=claim_object,
        allowed_parts=allowed_parts_str,
        allowed_issues=allowed_issues_str,
        conversation=conversation
    )
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    
    # Increment usage metrics
    API_USAGE["calls"] += 1
    if response.usage_metadata:
        API_USAGE["prompt_tokens"] += response.usage_metadata.prompt_token_count
        API_USAGE["completion_tokens"] += response.usage_metadata.candidates_token_count
    
    text = response.text.strip()
    try:
        data = parse_json_safely(text)
        claimed_issue = str(data.get("claimed_issue", "unknown")).lower().strip()
        claimed_part = str(data.get("claimed_part", "unknown")).lower().strip()
        
        # Verify lists
        if claimed_issue not in ALLOWED_ISSUES:
            claimed_issue = "unknown"
        if claimed_part not in parts:
            claimed_part = "unknown"
            
        result = {
            "claimed_issue": claimed_issue,
            "claimed_part": claimed_part
        }
        # Save to cache
        set_cached_response(cache_key, result)
        return result
    except Exception as e:
        logging.error(f"Failed to parse or map claim extractor output. Response: '{text}'. Error: {e}")
        return {
            "claimed_issue": "unknown",
            "claimed_part": "unknown"
        }