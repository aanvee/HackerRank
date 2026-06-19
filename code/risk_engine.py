def build_risk_flags(vision_result):

    flags = vision_result.get("risk_flags", [])

    if not vision_result["valid_image"]:
        flags.append("manual_review_required")

    return list(set(flags))