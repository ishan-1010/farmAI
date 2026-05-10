import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

TEXT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an expert agricultural extension officer helping small-scale farmers in India.
Given a crop problem (disease OR pest infestation), provide a practical, affordable treatment plan.
- For DISEASES (fungal/bacterial/viral): recommend fungicide in chemical_treatment.pesticide
- For PESTS (insects, worms like Fall Armyworm, bollworm): recommend insecticide in chemical_treatment.pesticide
- For WORMS specifically: include pheromone traps and biological controls (Bt spray) in organic_treatment
Return ONLY valid JSON with exactly these keys:
- immediate_action: most urgent step to take right now (string)
- organic_treatment: object with keys: method (string), frequency (string), preparation (string)
- chemical_treatment: object with keys: pesticide (string), dosage (string), frequency (string)
- prevention: list of 3-5 prevention tips (list of strings)
- estimated_recovery_days: estimated days to recovery (integer)
If the plant is healthy, recommend preventive care only.
Do not include any text outside the JSON object."""


def get_treatment(disease_name: str, severity: int, crop_name: str, problem_type: str = "disease") -> dict:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    user_message = (
        f"Crop: {crop_name}\n"
        f"Problem Type: {problem_type}\n"
        f"Problem: {disease_name}\n"
        f"Severity: {severity}/5\n"
        f"Provide treatment plan. Return ONLY valid JSON."
    )

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)

    # Pick sensible defaults based on problem type
    if problem_type == "pest":
        default_organic = {"method": "Neem oil + Bt spray", "frequency": "Every 5 days", "preparation": "5ml neem oil + 2g Bt per 1L water"}
        default_chemical = {"pesticide": "Emamectin Benzoate 5% SG", "dosage": "0.4g per litre of water", "frequency": "Every 7-10 days"}
    else:
        default_organic = {"method": "Neem oil spray", "frequency": "Weekly", "preparation": "5ml neem oil per 1L water"}
        default_chemical = {"pesticide": "Consult local agronomist", "dosage": "As directed", "frequency": "As directed"}

    result.setdefault("immediate_action", "Monitor the plant closely.")
    result.setdefault("organic_treatment", default_organic)
    result.setdefault("chemical_treatment", default_chemical)
    result.setdefault("prevention", ["Maintain proper spacing", "Avoid overwatering", "Remove infected leaves promptly"])
    result.setdefault("estimated_recovery_days", 14)
    return result
