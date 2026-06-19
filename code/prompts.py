CLAIM_EXTRACTION_PROMPT = """
You are extracting a damage claim.

Analyze the conversation and identify:

1. claimed issue type
2. claimed object part

Return ONLY valid JSON:

{
  "claimed_issue":"",
  "claimed_part":""
}

Allowed issue values:
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
"""