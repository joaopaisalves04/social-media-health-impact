import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Media Health Impact",
    page_icon="📱",
    layout="centered"
)

# ── Auto-train if model_bundle.pkl doesn't exist (e.g. on Streamlit Cloud) ───
if not os.path.exists("model_bundle.pkl"):
    with st.spinner("First run — training model, please wait..."):
        import train_and_save_model  # runs the training script

# ── Load model bundle (cached — loaded once per app instance) ─────────────────
@st.cache_resource
def load_bundle():
    return joblib.load("model_bundle.pkl")

bundle = load_bundle()

LABEL_COLORS = {
    "Negative": "#E74C3C",
    "Neutral":  "#F39C12",
    "Positive": "#2ECC71"
}
LABEL_EMOJIS = {
    "Negative": "🔴",
    "Neutral":  "🟡",
    "Positive": "🟢"
}

# ── Feature engineering (must match train_and_save_model.py exactly) ──────────
def preprocess_input(raw: dict) -> pd.DataFrame:
    median_usage  = bundle["median_usage"]
    top_platforms = list(bundle["top_platforms"])
    top_countries = list(bundle["top_countries"])
    feature_cols  = bundle["feature_columns"]

    row = {}

    # Numeric passthrough
    row["Age"]                   = raw["age"]
    row["Avg_Daily_Usage_Hours"] = raw["usage_hours"]
    row["Sleep_Hours_Per_Night"] = raw["sleep_hours"]
    row["Mental_Health_Score"]   = raw["mental_health"]

    # Binary / ordinal encodings
    row["Gender_enc"]    = 1 if raw["gender"] == "Female" else 0
    row["AcadPerf_enc"]  = 1 if raw["affects_academic"] == "Yes" else 0
    row["AcadLevel_enc"] = {"High School": 0, "Undergraduate": 1, "Graduate": 2}[raw["academic_level"]]

    # Engineered features
    row["UsageSleep_ratio"]    = raw["usage_hours"] / (raw["sleep_hours"] + 1e-5)
    row["HealthSleep_product"] = raw["mental_health"] * raw["sleep_hours"]
    row["HighUsage"]           = 1 if raw["usage_hours"] > median_usage else 0

    # Platform one-hot
    platform = raw["platform"] if raw["platform"] in top_platforms else "Other"
    for col in [c for c in feature_cols if c.startswith("plt_")]:
        row[col] = 1 if platform == col.replace("plt_", "") else 0

    # Country one-hot
    country = raw["country"] if raw["country"] in top_countries else "Other"
    for col in [c for c in feature_cols if c.startswith("ctry_")]:
        row[col] = 1 if country == col.replace("ctry_", "") else 0

    # Enforce exact column order from training
    return pd.DataFrame([row])[feature_cols]

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📱 Social Media Health Impact Predictor")
st.markdown(
    "Fill in your profile and click **Predict** to see whether social media "
    "use is likely having a **Negative**, **Neutral**, or **Positive** effect on your health."
)
st.divider()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Info")
        age           = st.slider("Age", min_value=18, max_value=24, value=20)
        gender        = st.radio("Gender", ["Male", "Female"], horizontal=True)
        academic_level = st.selectbox(
            "Academic Level", ["High School", "Undergraduate", "Graduate"]
        )
        affects_academic = st.radio(
            "Does social media affect your academic performance?",
            ["No", "Yes"],
            horizontal=True
        )

    with col2:
        st.subheader("Usage & Health")
        usage_hours   = st.slider("Avg Daily Usage (hrs)", 1.5, 8.5, 4.0, step=0.1)
        sleep_hours   = st.slider("Sleep Per Night (hrs)", 3.8, 9.6, 7.0, step=0.1)
        mental_health = st.slider("Mental Health Score (1–10)", 4.0, 9.0, 6.5, step=0.1)
        platform = st.selectbox(
            "Primary Platform",
            ["Instagram", "TikTok", "Facebook", "LinkedIn", "Twitter", "YouTube", "Snapchat", "Other"]
        )
        country = st.selectbox(
            "Country",
            ["India", "USA", "Canada", "Australia", "Other (not listed)"]
        )

    submitted = st.form_submit_button("Predict Impact", type="primary", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    # Normalise country "Other (not listed)" → "Other" for the model
    country_val = "Other" if country == "Other (not listed)" else country

    raw = dict(
        age=age,
        usage_hours=usage_hours,
        sleep_hours=sleep_hours,
        mental_health=mental_health,
        gender=gender,
        affects_academic=affects_academic,
        academic_level=academic_level,
        platform=platform,
        country=country_val
    )

    X_input = preprocess_input(raw)
    pred_idx   = bundle["model"].predict(X_input.values)[0]
    pred_label = bundle["label_map_inv"][pred_idx]
    proba      = bundle["model"].predict_proba(X_input.values)[0]

    st.divider()
    color = LABEL_COLORS[pred_label]
    emoji = LABEL_EMOJIS[pred_label]

    st.markdown(
        f"### Prediction: "
        f"<span style='color:{color}; font-size:1.4em'>{emoji} {pred_label}</span>",
        unsafe_allow_html=True
    )
    st.markdown("**Class probabilities:**")
    for idx, label in bundle["label_map_inv"].items():
        pct = float(proba[idx])
        st.progress(pct, text=f"{LABEL_EMOJIS[label]} {label}: {pct*100:.1f}%")
