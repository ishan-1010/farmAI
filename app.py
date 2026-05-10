import os
from pathlib import Path
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from agents.identifier import identify_crop
from agents.diagnoser import diagnose_disease
from agents.treatment import get_treatment
from agents.market import get_market_advice

load_dotenv()

st.set_page_config(
    page_title="FarmAI — Crop Doctor",
    page_icon="🌾",
    layout="wide",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2d6a4f, #52b788);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: #f8f9fa;
        border-left: 4px solid #52b788;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .severity-badge-1 { background:#d4edda; color:#155724; padding:4px 12px; border-radius:20px; font-weight:bold; }
    .severity-badge-2 { background:#d4edda; color:#155724; padding:4px 12px; border-radius:20px; font-weight:bold; }
    .severity-badge-3 { background:#fff3cd; color:#856404; padding:4px 12px; border-radius:20px; font-weight:bold; }
    .severity-badge-4 { background:#f8d7da; color:#721c24; padding:4px 12px; border-radius:20px; font-weight:bold; }
    .severity-badge-5 { background:#f8d7da; color:#721c24; padding:4px 12px; border-radius:20px; font-weight:bold; }
    .sell-now  { background:#dc3545; color:white; padding:8px 20px; border-radius:8px; font-size:1.2rem; font-weight:bold; }
    .wait      { background:#fd7e14; color:white; padding:8px 20px; border-radius:8px; font-size:1.2rem; font-weight:bold; }
    .process   { background:#6f42c1; color:white; padding:8px 20px; border-radius:8px; font-size:1.2rem; font-weight:bold; }
    .consult   { background:#6c757d; color:white; padding:8px 20px; border-radius:8px; font-size:1.2rem; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>FarmAI — Crop Doctor</h1>
    <p>AI-powered crop disease diagnosis & market advisory for farmers</p>
    <small>Powered by Llama 4 Scout & Llama 3.3 70B via Groq</small>
</div>
""", unsafe_allow_html=True)

# ── API Key check ─────────────────────────────────────────────────────────────
if not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not set. Add it to your .env file or environment.")
    st.code("GROQ_API_KEY=your_key_here", language="bash")
    st.stop()

# ── Demo Sidebar ──────────────────────────────────────────────────────────────
DEMO_DIR = Path(__file__).parent / "demo_images"
DEMO_OPTIONS = {
    "🍅  Tomato — Late Blight (Disease)":     "tomato_late_blight.jpg",
    "🌽  Maize — Gray Leaf Spot (Disease)":   "corn_gray_leaf_spot.jpg",
    "🌽  Maize — Fall Armyworm (Pest)":       "fall_armyworm_maize.jpg",
    "🥔  Potato — Late Blight (Disease)":     "potato_late_blight.jpg",
    "🌾  Wheat — Leaf Rust (Disease)":        "wheat_leaf_rust.jpg",
}

with st.sidebar:
    st.markdown("### 🎯 Demo Mode")
    st.caption("Pick a sample image to see all 4 agents in action")
    demo_choice = st.radio("Sample crop images:", ["— Upload my own —"] + list(DEMO_OPTIONS.keys()))
    if demo_choice != "— Upload my own —":
        demo_path = DEMO_DIR / DEMO_OPTIONS[demo_choice]
        if demo_path.exists():
            st.image(str(demo_path), width="stretch")
    st.divider()
    st.markdown("**Models used:**")
    st.caption("• Llama 4 Scout (Vision) — Agents 1 & 2")
    st.caption("• Llama 3.3 70B — Agents 3 & 4")
    st.caption("• Groq free tier — no cost")

# ── Input Section ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

# Resolve image source: demo pick or manual upload
demo_image_bytes = None
if demo_choice != "— Upload my own —":
    demo_path = DEMO_DIR / DEMO_OPTIONS[demo_choice]
    if demo_path.exists():
        demo_image_bytes = demo_path.read_bytes()

with col1:
    st.subheader("Crop Image")
    if demo_image_bytes:
        st.image(demo_image_bytes, caption=demo_choice, width="stretch")
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader(
            "Take a clear photo of the affected leaf or plant",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded image", width="stretch")

with col2:
    st.subheader("Additional Info (Optional)")
    symptoms_text = st.text_area(
        "Describe what you observe",
        placeholder="e.g. Yellow spots on leaves, brown edges, wilting stems...",
        height=120,
    )
    farmer_state = st.selectbox(
        "Your State (for market prices)",
        ["Maharashtra", "Punjab", "Uttar Pradesh", "Karnataka", "Gujarat",
         "Haryana", "Madhya Pradesh", "Andhra Pradesh", "Tamil Nadu", "Telangana", "Other"],
    )

analyze_btn = st.button("🔬 Analyze My Crop", type="primary")

# ── Analysis Pipeline ─────────────────────────────────────────────────────────
image_bytes = demo_image_bytes or (uploaded_file.getvalue() if uploaded_file else None)

if analyze_btn and image_bytes:
    results = {}

    st.divider()
    st.subheader("Analysis in Progress")

    # Agent 1 — Crop Identifier
    with st.spinner("Agent 1: Identifying crop type..."):
        try:
            results["identify"] = identify_crop(image_bytes)
            crop_name = results["identify"]["crop_name"]
            confidence = results["identify"]["confidence"]
            st.success(f"✅ **Crop Identified:** {crop_name} — Confidence: {confidence.upper()}")
        except Exception as e:
            st.error(f"Agent 1 failed: {e}")
            st.stop()

    # Agent 2 — Disease Diagnoser
    with st.spinner("Agent 2: Diagnosing disease..."):
        try:
            results["diagnose"] = diagnose_disease(image_bytes, crop_name)
            disease = results["diagnose"]["disease_name"]
            severity = results["diagnose"]["severity"]
            is_healthy = results["diagnose"]["is_healthy"]
            if is_healthy:
                st.success("✅ **Plant is Healthy!** No disease detected.")
            else:
                st.warning(f"⚠️ **Disease Detected:** {disease} — Severity: {severity}/5")
        except Exception as e:
            st.error(f"Agent 2 failed: {e}")
            st.stop()

    # Agent 3 — Treatment Advisor
    with st.spinner("Agent 3: Building treatment plan..."):
        try:
            results["treatment"] = get_treatment(
                results["diagnose"]["disease_name"],
                results["diagnose"]["severity"],
                crop_name,
                results["diagnose"].get("problem_type", "disease"),
            )
            st.success("✅ **Treatment Plan Ready**")
        except Exception as e:
            st.error(f"Agent 3 failed: {e}")
            st.stop()

    # Agent 4 — Market Intelligence
    with st.spinner("Agent 4: Checking mandi prices..."):
        try:
            results["market"] = get_market_advice(
                crop_name,
                results["diagnose"]["severity"],
                results["treatment"]["estimated_recovery_days"],
                disease_name=results["diagnose"]["disease_name"],
            )
            price = results["market"]["modal_price"]
            rec = results["market"]["recommendation"]
            st.success(f"✅ **Market Price:** ₹{price} {results['market']['price_unit']} — {rec}")
        except Exception as e:
            st.error(f"Agent 4 failed: {e}")
            st.stop()

    # ── Full Report ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Full Diagnostic Report")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Crop", results["identify"]["crop_name"])
    r2.metric("Disease", results["diagnose"]["disease_name"])
    r3.metric("Severity", f"{results['diagnose']['severity']}/5")
    r4.metric("Recovery", f"~{results['treatment']['estimated_recovery_days']} days")

    tab1, tab2, tab3, tab4 = st.tabs(["🌿 Crop Info", "🦠 Disease", "💊 Treatment", "📈 Market"])

    with tab1:
        id_r = results["identify"]
        st.markdown(f"**Common Name:** {id_r['crop_name']}")
        st.markdown(f"**Scientific Name:** *{id_r['scientific_name']}*")
        st.markdown(f"**Identification Confidence:** {id_r['confidence'].upper()}")
        if id_r["notes"]:
            st.info(id_r["notes"])

    with tab2:
        diag = results["diagnose"]
        if diag["is_healthy"]:
            st.success("Your plant is healthy! No disease detected.")
        else:
            st.markdown(f"**Disease:** {diag['disease_name']}")
            st.markdown(f"**Pathogen:** *{diag['pathogen']}*")
            severity_css = f"severity-badge-{diag['severity']}"
            st.markdown(f"**Severity:** <span class='{severity_css}'>{diag['severity_label']} ({diag['severity']}/5)</span>", unsafe_allow_html=True)
            st.markdown(f"**Affected Area:** ~{diag['affected_area_percent']}%")
            if diag["symptoms_observed"]:
                st.markdown("**Symptoms Observed:**")
                for s in diag["symptoms_observed"]:
                    st.markdown(f"  - {s}")

    with tab3:
        treat = results["treatment"]
        st.error(f"**Immediate Action:** {treat['immediate_action']}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 🌱 Organic Treatment")
            org = treat["organic_treatment"]
            st.markdown(f"**Method:** {org['method']}")
            st.markdown(f"**Frequency:** {org['frequency']}")
            st.markdown(f"**Preparation:** {org['preparation']}")

        with col_b:
            st.markdown("#### ⚗️ Chemical Treatment")
            chem = treat["chemical_treatment"]
            label = "Insecticide" if results["diagnose"].get("problem_type") == "pest" else "Fungicide/Pesticide"
            st.markdown(f"**{label}:** {chem.get('pesticide', chem.get('fungicide', 'N/A'))}")
            st.markdown(f"**Dosage:** {chem['dosage']}")
            st.markdown(f"**Frequency:** {chem['frequency']}")

        st.markdown("#### Prevention Tips")
        for tip in treat["prevention"]:
            st.markdown(f"- {tip}")

    with tab4:
        mkt = results["market"]
        rec = mkt["recommendation"]
        css_class = {"SELL NOW": "sell-now", "WAIT": "wait", "PROCESS LOCALLY": "process"}.get(rec, "consult")
        st.markdown(f"<div class='{css_class}'>{rec}</div><br>", unsafe_allow_html=True)
        if mkt.get("inferred"):
            st.caption(f"ℹ️ Crop was unidentified — market data shown for **{mkt['crop']}** (inferred from disease)")

        m1, m2, m3 = st.columns(3)
        m1.metric("Modal Price", f"₹{mkt['modal_price']}", help=mkt["price_unit"])
        m2.metric("Market", mkt["market"])
        m3.metric("Trend", mkt["price_trend"].capitalize())

        st.info(mkt["reasoning"])
        st.caption(f"Last updated: {mkt['last_updated']} | Source: Agmarknet Mandi Data")

elif analyze_btn and not uploaded_file:
    st.warning("Please upload a crop image to proceed.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("FarmAI | Built for AI for Social Good Hackathon | Powered by Groq + Llama 4 Scout (Open Source)")
