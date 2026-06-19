import json
from PIL import Image

def analyze_image(client, image_path, claim_object):

    image = Image.open(image_path)

    prompt = f"""
You are an insurance claim verification expert.

Analyze ALL provided images together.

Return ONLY valid JSON.

{{
  "issue_type":"",
  "object_part":"",
  "severity":"",
  "valid_image":true,
  "risk_flags":[],
  "evidence_standard_met":true,
  "evidence_standard_met_reason":"",
  "supporting_image_ids":[],
  "claim_status_suggestion":"",
  "claim_status_justification":""
}}

Allowed issue_type values:
dent
scratch
crack
glass_shatter
broken_part
missing_part
torn_packaging
crushed_packaging
water_damage
stain
none
unknown

Allowed severity values:
none
low
medium
high
unknown

Allowed claim_status_suggestion values:
supported
contradicted
not_enough_information

Allowed risk_flags:
blurry_image
cropped_or_obstructed
low_light_or_glare
wrong_angle
wrong_object
wrong_object_part
damage_not_visible
claim_mismatch
possible_manipulation
non_original_image
text_instruction_present
user_history_risk
manual_review_required

Use the images as primary evidence.

Return JSON only.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, image]
    )

    text = response.text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)