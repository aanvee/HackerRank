import logging
from utils import ALLOWED_STATUSES

def generate_claim_status(
    claimed_issue,
    claimed_part,
    vision_result,
    risk_flags
):
    """
    Combines VLM outputs with programmatic rule layers to compute final claim status.
    """
    vlm_status = str(vision_result.get("claim_status_suggestion", "")).lower().strip()
    evidence_standard_met = vision_result.get("evidence_standard_met", True)
    valid_image = vision_result.get("valid_image", True)
    detected_issue = vision_result.get("issue_type", "unknown")
    detected_part = vision_result.get("object_part", "unknown")
    
    status = vlm_status
    if status not in ALLOWED_STATUSES:
        status = "not_enough_information"

    # Rule checks & overrides
    if not valid_image:
        if "wrong_object" in risk_flags or "claim_mismatch" in risk_flags:
            status = "contradicted"
        else:
            status = "not_enough_information"

    if not evidence_standard_met:
        if "wrong_object" in risk_flags:
            status = "contradicted"
        else:
            status = "not_enough_information"

    if "wrong_object" in risk_flags:
        status = "contradicted"
        
    if "damage_not_visible" in risk_flags:
        # Part is visible but damage is not visible
        if "wrong_angle" in risk_flags or "cropped_or_obstructed" in risk_flags:
            status = "not_enough_information"
        else:
            status = "contradicted"

    if "wrong_angle" in risk_flags or "cropped_or_obstructed" in risk_flags:
        if status != "supported":
            status = "not_enough_information"

    if detected_issue == "none" and claimed_issue not in ["none", "unknown"]:
        if "wrong_angle" not in risk_flags and "cropped_or_obstructed" not in risk_flags:
            status = "contradicted"

    if "claim_mismatch" in risk_flags:
        status = "contradicted"

    if status not in ALLOWED_STATUSES:
        status = "not_enough_information"

    # Align justification
    justification = vision_result.get("claim_status_justification", "").strip()
    
    # Clean up justification
    if not justification:
        if status == "supported":
            justification = f"The visual evidence shows a visible {detected_issue} on the {detected_part} supporting the claim."
        elif status == "contradicted":
            justification = f"The visual evidence contradicts the claim of {claimed_issue} on the {claimed_part}."
        else:
            justification = "The submitted images do not provide enough clear information to evaluate the claim."
            
    if "user_history_risk" in risk_flags and "user history" not in justification.lower():
        justification += " Note: User claim history presents risk flags requiring attention."

    return status, justification