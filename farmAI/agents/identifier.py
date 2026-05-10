import os
import json
import base64
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

SYSTEM_PROMPT = """You are an expert agricultural botanist with 20+ years of field experience.
Analyze the plant image and identify the crop type.
Return ONLY valid JSON with exactly these keys:
- crop_name: common name of the crop (string, use "Unknown" if cannot identify)
- scientific_name: Latin binomial name (string, use "Unknown" if cannot identify)
- confidence: your confidence level (string: "high", "medium", or "low")
- notes: brief observation about identifying features (string)
Do not include any text outside the JSON object."""


def encode_image(image_bytes: bytes) -> str:
    return base64.standard_b64encode(image_bytes).decode("utf-8")


def identify_crop(image_bytes: bytes) -> dict:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    b64 = encode_image(image_bytes)

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": "Identify this crop. Return ONLY valid JSON.",
                    },
                ],
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)

    # Ensure required keys with safe defaults
    result.setdefault("crop_name", "Unknown")
    result.setdefault("scientific_name", "Unknown")
    result.setdefault("confidence", "low")
    result.setdefault("notes", "")
    return result
