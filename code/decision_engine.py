def generate_claim_status(
    claimed_issue,
    detected_issue,
    risk_flags
):

    if (
        "wrong_angle" in risk_flags
        or "cropped_or_obstructed" in risk_flags
    ):
        return "not_enough_information"

    if detected_issue == "unknown":
        return "not_enough_information"

    if claimed_issue == detected_issue:
        return "supported"

    return "contradicted"