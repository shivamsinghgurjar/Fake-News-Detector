import streamlit as st
import requests
import pickle
import numpy as np
import os

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# ─────────────────────────────────────────────
# SIMPLE WHITE CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* White background, clean look */
    .stApp { background-color: #ffffff; }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 4px;
    }
    .sub-title {
        font-size: 1rem;
        color: #666666;
        margin-bottom: 32px;
    }
    .result-card {
        padding: 16px 20px;
        border-radius: 10px;
        margin: 6px 0;
        font-size: 1rem;
        font-weight: 600;
    }
    .real  { background: #e6f9f0; color: #1a7a4a; border-left: 4px solid #27ae60; }
    .fake  { background: #fdecea; color: #a93226; border-left: 4px solid #e74c3c; }
    .info  { background: #eaf3fc; color: #1a5276; border-left: 4px solid #2980b9; }
    .divider { border: none; border-top: 1px solid #eeeeee; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}

    ml_path = "models/best_model.pkl"
    if os.path.exists(ml_path):
        with open(ml_path, "rb") as f:
            models["ml"] = pickle.load(f)

    tok_path = "models/tokenizer.pkl"
    if os.path.exists(tok_path):
        with open(tok_path, "rb") as f:
            models["tokenizer"] = pickle.load(f)

    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        ann_path = "models/ann_model.h5"
        rnn_path = "models/rnn_model.h5"

        if os.path.exists(ann_path):
            models["ann"] = load_model(ann_path)
        if os.path.exists(rnn_path):
            models["rnn"] = load_model(rnn_path)

        models["pad_sequences"] = pad_sequences
    except Exception:
        pass

    return models


# ─────────────────────────────────────────────
# FACT CHECK API
# ─────────────────────────────────────────────
API_KEY = "AIzaSyBoCfO2wRcNlva1z5rXXhYH6_oxjc8X18k"


def fact_check_api(query: str) -> str:
    try:
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        res = requests.get(url, params={"query": query, "key": API_KEY}, timeout=5).json()
        claims = res.get("claims", [])
        if not claims:
            return "No fact-check found"
        review = claims[0]['claimReview'][0]
        return review['publisher']['name'] + ": " + review['textualRating']
    except Exception:
        return "API unavailable"


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🔍 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">ML + Deep Learning + Google Fact Check API</div>', unsafe_allow_html=True)

models = load_models()

user_input = st.text_area(
    "Enter a news statement:",
    placeholder="e.g. Vaccines contain microchips.",
    height=130
)

analyze = st.button("Analyze", type="primary", use_container_width=True)

if analyze:
    if not user_input.strip():
        st.warning("Please enter a statement.")
    else:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        with st.spinner("Analyzing..."):

            # ── ML Prediction ──────────────────────────────
            if "ml" in models:
                ml_pred = models["ml"].predict([user_input])[0]
                ml_proba = models["ml"].predict_proba([user_input])[0]
                ml_label = "Real" if ml_pred == 1 else "Fake"
                ml_conf = round(max(ml_proba) * 100, 1)
                css = "real" if ml_label == "Real" else "fake"
                icon = "✅" if ml_label == "Real" else "❌"
                st.markdown(
                    f'<div class="result-card {css}">{icon} ML Model — {ml_label} &nbsp;'
                    f'<span style="font-weight:400;font-size:0.9rem;">({ml_conf}% confidence)</span></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown('<div class="result-card info">⚠️ ML model not found. Run pipeline.py first.</div>',
                            unsafe_allow_html=True)

            # ── ANN Prediction ─────────────────────────────
            if "ann" in models and "tokenizer" in models:
                pad = models["pad_sequences"](
                    models["tokenizer"].texts_to_sequences([user_input]), maxlen=100
                )
                ann_score = models["ann"].predict(pad, verbose=0)[0][0]
                ann_label = "Real" if ann_score > 0.5 else "Fake"
                css = "real" if ann_label == "Real" else "fake"
                icon = "✅" if ann_label == "Real" else "❌"
                st.markdown(
                    f'<div class="result-card {css}">{icon} ANN Model — {ann_label} &nbsp;'
                    f'<span style="font-weight:400;font-size:0.9rem;">({round(float(ann_score)*100,1)}% real score)</span></div>',
                    unsafe_allow_html=True
                )

            # ── RNN Prediction ─────────────────────────────
            if "rnn" in models and "tokenizer" in models:
                rnn_score = models["rnn"].predict(pad, verbose=0)[0][0]
                rnn_label = "Real" if rnn_score > 0.5 else "Fake"
                css = "real" if rnn_label == "Real" else "fake"
                icon = "✅" if rnn_label == "Real" else "❌"
                st.markdown(
                    f'<div class="result-card {css}">{icon} RNN (LSTM) — {rnn_label} &nbsp;'
                    f'<span style="font-weight:400;font-size:0.9rem;">({round(float(rnn_score)*100,1)}% real score)</span></div>',
                    unsafe_allow_html=True
                )

            # ── Fact Check API ─────────────────────────────
            api_result = fact_check_api(user_input)
            st.markdown(
                f'<div class="result-card info">🌐 Fact Check API — {api_result}</div>',
                unsafe_allow_html=True
            )