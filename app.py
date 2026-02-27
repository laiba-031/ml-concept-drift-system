import os
import streamlit as st
import pandas as pd
import joblib

from src.train import train_and_evaluate
from src.drift_detection import run_drift_detection

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "churn.csv")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ML Drift System",
    page_icon="📉",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.metric-card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
.big-number {
    font-size: 32px;
    font-weight: bold;
}
.sub-text {
    color: #8b949e;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("📉 ML Concept Drift Monitoring System")
st.caption("A production-style ML system with monitoring & retraining")

# ---------------- LOAD MODEL ----------------
if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Please train the model first.")
    st.stop()

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# ---------------- TOP METRICS ----------------
st.markdown("### 🔍 System Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="big-number">{df.shape[0]}</div>
            <div class="sub-text">Total Records</div>
        </div>
        """, unsafe_allow_html=True
    )

with col2:
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="big-number">{churn_rate:.1f}%</div>
            <div class="sub-text">Churn Rate</div>
        </div>
        """, unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="big-number">Active</div>
            <div class="sub-text">Model Status</div>
        </div>
        """, unsafe_allow_html=True
    )

st.divider()

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(
    ["🎯 Prediction", "⚠️ Drift Monitoring", "🔁 Retraining"]
)

# ---------------- TAB 1: PREDICTION ----------------
with tab1:
    st.subheader("Customer Churn Prediction")

    df_input = df.drop("customerID", axis=1)

    for col in df_input.select_dtypes(include="object").columns:
        if col != "Churn":
            df_input[col] = df_input[col].astype("category").cat.codes

    X = df_input.drop("Churn", axis=1)
    preds = model.predict_proba(X)[:, 1]

    result_df = pd.DataFrame({
        "Customer Index": range(1, 11),
        "Churn Probability": preds[:10]
    })

    st.dataframe(result_df, use_container_width=True)

# ---------------- TAB 2: DRIFT ----------------
with tab2:
    st.subheader("Concept Drift Detection")

    st.write(
        "This module checks whether incoming data distribution "
        "has shifted significantly from training data."
    )

    if st.button("🔍 Check Drift"):
        with st.spinner("Analyzing feature distributions..."):
            drifted = run_drift_detection()

        if drifted:
            st.error(f"Drift detected in features: {drifted}")
        else:
            st.success("No drift detected. Data is stable.")

# ---------------- TAB 3: RETRAIN ----------------
with tab3:
    st.subheader("Model Retraining")

    st.write(
        "Retraining is triggered only if the new model "
        "outperforms the existing one."
    )

    if st.button("🔄 Retrain Model"):
        with st.spinner("Retraining model..."):
            result = train_and_evaluate()
        st.success(result)

# ---------------- FOOTER ----------------
st.divider()
st.caption("Built as an end-to-end ML system with monitoring and automation")