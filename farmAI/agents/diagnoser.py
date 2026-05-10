import os
import json
import base64
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

SYSTEM_PROMPT_TEMPLATE = """You are an expert crop protection specialist for {crop_name}.
Carefully examine the image for ANY problem: fungal/bacterial/viral diseases, insect pests, worm infestations, or physical damage.
Return ONLY valid JSON with exactly these keys:
- problem_type: one of "disease", "pest", "healthy" (string)
- disease_name: name of the disease or pest (e.g. "Fall Armyworm", "Early Blight"). Use "None" if healthy.
- pathogen: causative organism or pest species (e.g. "Spodoptera frugiperda", "Alternaria solani"). Use "None" if healthy.
- severity: integer from 1 (very mild) to 5 (critical), use 0 if healthy
- severity_label: one of "Healthy", "Mild", "Moderate", "Severe", "Critical"
- affected_area_percent: estimated percentage of plant affected (integer 0-100)
- symptoms_observed: list of visible symptoms such as leaf holes, frass, egg masses, spots, lesions (list of strings)
- is_healthy: true if plant is healthy, false otherwise (boolean)
Do not include any text outside the JSON object."""

SEVERITY_LABELS = {0: "Healthy", 1: "Mild", 2: "Mild", 3: "Moderate", 4: "Severe", 5: "Critical"}


def diagnose_disease(image_bytes: bytes, crop_name: str) -> dict:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(crop_name=crop_name or "Unknown crop")

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": f"Diagnose this {crop_name} plant. Return ONLY valid JSON.",
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)

    # Sanitize severity to valid range
    severity = result.get("severity", 0)
    if not isinstance(severity, int) or severity < 0 or severity > 5:
        severity = 0
    result["severity"] = severity
    result["severity_label"] = SEVERITY_LABELS.get(severity, "Unknown")

    result.setdefault("problem_type", "healthy" if severity == 0 else "disease")
    result.setdefault("disease_name", "None")
    result.setdefault("pathogen", "None")
    result.setdefault("affected_area_percent", 0)
    result.setdefault("symptoms_observed", [])
    result.setdefault("is_healthy", severity == 0)
    return result
