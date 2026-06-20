from google import genai
from dotenv import load_dotenv
from pathlib import Path
import os
from PIL import Image

load_dotenv(Path(__file__).parent.parent / ".env")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

try:
    # Test simple text gen
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Hello, say hello back!"
    )
    print("Text test:", response.text.strip())

    # Test image gen
    img_path = Path(__file__).parent.parent / "dataset/images/sample/case_001/img_1.jpg"
    if img_path.exists():
        img = Image.open(img_path)
        response_img = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=["Describe this image in one word.", img]
        )
        print("Image test:", response_img.text.strip())
    else:
        print("Image not found at:", img_path)
except Exception as e:
    print("Error:", e)