from claim_extractor import extract_claim
from image_analyzer import analyze_image
from decision_engine import generate_claim_status
from risk_engine import build_risk_flags


def process_claim(client, row):

    claim = extract_claim(
        client,
        row["user_claim"]
    )

    first_image = row["image_paths"].split(";")[0]

    vision = analyze_image(
        client,
        "../" + first_image,
        row["claim_object"]
    )

    risk_flags = build_risk_flags(vision)

    status = generate_claim_status(
        claim["claimed_issue"],
        vision["visible_issue"],
        risk_flags
    )

    return {
        "issue_type": vision["visible_issue"],
        "object_part": vision["visible_part"],
        "claim_status": status,
        "risk_flags": ";".join(risk_flags),
        "severity": vision["severity"]
    }