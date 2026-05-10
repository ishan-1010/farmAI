import os
import json
import pandas as pd
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

TEXT_MODEL = "llama-3.3-70b-versatile"
CSV_PATH = Path(__file__).parent.parent / "data" / "mandi_prices.csv"

SYSTEM_PROMPT = """You are an agricultural market analyst advising Indian farmers.
Given crop price data and disease status, provide a concise market recommendation.
Return ONLY valid JSON with exactly these keys:
- crop: crop name (string)
- modal_price: modal market price as integer (integer)
- price_unit: unit description like "per Quintal" (string)
- market: market name and state (string)
- last_updated: date string (string)
- recommendation: one of "SELL NOW", "WAIT", or "PROCESS LOCALLY" (string)
- reasoning: 1-2 sentence explanation of recommendation (string)
- price_trend: one of "rising", "stable", or "falling" (string)
Do not include any text outside the JSON object."""


def _lookup_price(crop_name: str) -> dict | None:
    if not CSV_PATH.exists():
        return None
    df = pd.read_csv(CSV_PATH)
    match = df[df["crop"].str.lower() == crop_name.lower()]
    if match.empty:
        # Try partial match
        match = df[df["crop"].str.lower().str.contains(crop_name.lower(), na=False)]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        "crop": row["crop"],
        "modal_price": int(row["modal_price"]),
        "min_price": int(row["min_price"]),
        "max_price": int(row["max_price"]),
        "unit": row["unit"],
        "market": f"{row['market']}, {row['state']}",
        "last_updated": row["last_updated"],
    }


def get_market_advice(crop_name: str, severity: int, recovery_days: int) -> dict:
    price_data = _lookup_price(crop_name)

    if price_data is None:
        return {
            "crop": crop_name,
            "modal_price": 0,
            "price_unit": "Data not available",
            "market": "N/A",
            "last_updated": "N/A",
            "recommendation": "CONSULT LOCAL MANDI",
            "reasoning": f"No price data found for {crop_name}. Please check your local mandi.",
            "price_trend": "unknown",
        }

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    user_message = (
        f"Crop: {price_data['crop']}\n"
        f"Modal Price: ₹{price_data['modal_price']} per {price_data['unit']}\n"
        f"Price Range: ₹{price_data['min_price']} - ₹{price_data['max_price']}\n"
        f"Market: {price_data['market']}\n"
        f"Disease Severity: {severity}/5\n"
        f"Days to Recovery: {recovery_days}\n"
        f"Should the farmer sell now or wait? Return ONLY valid JSON."
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

    # Ensure price data is present from CSV (LLM may hallucinate prices)
    result["crop"] = price_data["crop"]
    result["modal_price"] = price_data["modal_price"]
    result["price_unit"] = f"per {price_data['unit']}"
    result["market"] = price_data["market"]
    result["last_updated"] = price_data["last_updated"]
    result.setdefault("recommendation", "CONSULT LOCAL MANDI")
    result.setdefault("reasoning", "Unable to determine recommendation.")
    result.setdefault("price_trend", "stable")
    return result
