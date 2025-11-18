import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import json
import os
import matplotlib.pyplot as plt
from fpdf import FPDF


# --------------------------------------------------------
# Load models and preprocessing pipeline
# --------------------------------------------------------
@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(__file__)
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
    model_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    return preprocessor, model


preprocessor, xgb_model = load_artifacts()


# --------------------------------------------------------
# SHAP Explainer
# --------------------------------------------------------
@st.cache_resource
def load_shap_explainer():
    explainer = shap.TreeExplainer(xgb_model)
    return explainer


shap_explainer = load_shap_explainer()


# --------------------------------------------------------
# Features
# --------------------------------------------------------
FEATURES = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]

HOME_OWNERSHIP = ["RENT", "OWN", "OTHER"]
LOAN_INTENT = ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]
LOAN_GRADE = ["A", "B", "C", "D", "E", "F", "G"]
DEFAULT_FLAG = ["Y", "N"]


# --------------------------------------------------------
# Prediction Logic
# --------------------------------------------------------
def predict_single(input_dict):
    df = pd.DataFrame([input_dict])
    X = preprocessor.transform(df)
    prob = float(xgb_model.predict_proba(X)[0][1])  # Cast to Python float
    pred = int(prob >= 0.5)
    return prob, pred, X


# --------------------------------------------------------
# PDF Report Generator
# --------------------------------------------------------
def generate_pdf_report(input_dict, prob, pred):

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, txt="Loan Default Prediction Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Prediction Summary", ln=True)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Default Probability:  {prob*100:.2f}%", ln=True)
    pdf.cell(0, 8, f"Prediction: {'HIGH RISK' if pred==1 else 'LOW RISK'}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Input Details", ln=True)

    pdf.set_font("Arial", size=12)
    for k, v in input_dict.items():
        pdf.cell(0, 8, f"{k}: {v}", ln=True)

    out_path = "loan_report.pdf"
    pdf.output(out_path)
    return out_path


# --------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------
st.set_page_config(
    page_title="Loan Default Predictor",
    layout="wide",
    page_icon="💳"
)

st.title("💳 Loan Default Prediction App")
st.write("Predict default probability, explain the model using SHAP, and generate reports.")


tab1, tab2, tab3 = st.tabs([
    "🔮 Predict Single Customer",
    "📂 Batch Prediction (CSV)",
    "📊 Model Explainability (SHAP)"
])


# --------------------------------------------------------
# TAB 1 — Single Customer Prediction
# --------------------------------------------------------
with tab1:
    st.header("🔹 Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        person_age = st.number_input("Age", 18, 100, 30)
        person_income = st.number_input("Annual Income", 1000, 5000000, 50000)
        person_emp_length = st.number_input("Employment Length (years)", 0, 50, 5)
        loan_amnt = st.number_input("Loan Amount", 500, 500000, 15000)
        loan_percent_income = st.number_input("Loan % of Income", 0.0, 2.0, 0.3, step=0.01)
        cb_person_cred_hist_length = st.number_input("Credit History Length", 0, 50, 3)

    with col2:
        person_home_ownership = st.selectbox("Home Ownership", HOME_OWNERSHIP)
        loan_intent = st.selectbox("Loan Intent", LOAN_INTENT)
        loan_grade = st.selectbox("Loan Grade", LOAN_GRADE)
        loan_int_rate = st.number_input("Interest Rate (%)", 1.0, 40.0, 12.5)
        cb_person_default_on_file = st.selectbox("Defaults on File?", DEFAULT_FLAG)

    if st.button("Predict Default Risk", use_container_width=True):

        input_dict = {
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
        }

        prob, pred, X_vec = predict_single(input_dict)

        st.subheader("📊 Prediction Result")

        colA, colB = st.columns(2)

        with colA:
            st.metric("Default Probability", f"{prob*100:.2f}%")

        with colB:
            if pred == 1:
                st.error("🚨 HIGH RISK of Default")
            else:
                st.success("🟢 LOW RISK of Default")

        # Fix progress bar (must be Python float)
        st.write("### Risk Gauge")
        st.progress(min(float(prob), 0.99))

        # PDF Download
        report_path = generate_pdf_report(input_dict, prob, pred)
        with open(report_path, "rb") as f:
            st.download_button(
                label="📄 Download PDF Report",
                data=f,
                file_name="loan_prediction_report.pdf",
                mime="application/pdf"
            )

        # SHAP Personal Explanation
        st.write("---")
        st.write("### 🔍 SHAP Explanation for this Prediction")

        shap_values = shap_explainer(X_vec)

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_values[0], max_display=12, show=False)
        st.pyplot(fig)


# --------------------------------------------------------
# TAB 2 — Batch CSV
# --------------------------------------------------------
with tab2:
    st.header("📂 Upload CSV for Bulk Prediction")
    st.code(", ".join(FEATURES))

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        try:
            X = preprocessor.transform(df)
            probs = xgb_model.predict_proba(X)[:, 1]
            df["prob_default"] = probs
            df["pred_default"] = (probs >= 0.5).astype(int)

            st.success("Batch prediction completed!")
            st.dataframe(df.head())

            st.download_button(
                "📥 Download Predictions CSV",
                df.to_csv(index=False).encode("utf-8"),
                "loan_predictions.csv",
                "text/csv"
            )
        except Exception as e:
            st.error(f"Error: {e}")


# --------------------------------------------------------
# TAB 3 — Global SHAP Explainability
# --------------------------------------------------------
with tab3:
    st.header("📊 Global Model Explainability (SHAP)")

    sample = st.slider("Select sample size:", 100, 2000, 500)

    BASE_DIR = os.path.dirname(__file__)
    X_TEST = os.path.join(BASE_DIR, "data", "processed", "X_test.npy")
    X_test = np.load(X_TEST)

    shap_vals = shap_explainer(X_test[:sample])

    st.subheader("📌 Feature Importance (SHAP Summary Plot)")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_test[:sample], show=False)
    st.pyplot(fig1)

    st.write("Higher SHAP value = greater effect on increasing or decreasing default risk.")
