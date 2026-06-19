import json
from google import genai
from prompts import CLAIM_EXTRACTION_PROMPT

def extract_claim(client, conversation):

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
{CLAIM_EXTRACTION_PROMPT}

Conversation:
{conversation}
"""
    )

    text = response.text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)