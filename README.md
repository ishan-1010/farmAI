# FarmAI — AI Crop Doctor

### Member Info: Ishan Katoch, Chayan, Ariti, Ramanpreet, Aditi, Gursheen
</br>

An open-source multi-agent AI application that helps farmers diagnose crop diseases and get market price advice from a single photo.

Built for the **AI for Social Good Hackathon** using 100% open-source LLMs via Groq.

## What It Does

Upload a photo of your crop → 4 AI agents work sequentially:

| Agent | Model | Task |
|---|---|---|
| Crop Identifier | Llama 4 Scout (Vision) | Identifies the crop type |
| Disease Diagnoser | Llama 4 Scout (Vision) | Detects disease & severity |
| Treatment Advisor | Llama 3.3 70B | Recommends organic & chemical treatment |
| Market Intelligence | Llama 3.3 70B + CSV | Gives mandi price & sell/hold advice |

## Setup

### 1. Get a free Groq API key
Sign up at [console.groq.com](https://console.groq.com) — free tier, no credit card needed.

### 2. Clone and install
```bash
git clone https://github.com/your-username/farmAI
cd farmAI
pip install -r requirements.txt
```

### 3. Set your API key
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 4. Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Run Tests
```bash
python -m pytest tests/ -v
```

All 25 micro tests run fully offline — no API key needed.

## Tech Stack
- **UI:** Streamlit
- **Vision AI:** Llama 4 Scout (`meta-llama/llama-4-scout-17b-16e-instruct`) via Groq
- **Text AI:** Llama 3.3 70B Versatile via Groq
- **Market Data:** Indian Mandi prices (Agmarknet-sourced CSV)
- **Language:** Python 3.11+

## Social Impact
- 600M+ farmers in India lack access to affordable agronomic advice
- Most crop losses are preventable with early disease detection
- This tool works on any laptop, offline market data, free AI tier

## Contributing
PRs welcome! See open issues for ideas.

## License
MIT
