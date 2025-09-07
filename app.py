# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

MODEL_PATH = "model_pipeline.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()
st.title("Loan Approval Predictor")
st.write("Enter applicant and loan details and click Predict. Model and preprocessing are saved in a single pipeline.")

# Sidebar or main inputs
st.header("Applicant details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male","Female","Other"], index=0)
    married = st.selectbox("Married", ["Yes","No"], index=1)
    dependents = st.selectbox("Dependents", ["0","1","2","3+"], index=0)
    education = st.selectbox("Education", ["Graduate","Not Graduate"], index=0)
    self_employed = st.selectbox("Self Employed", ["No","Yes"], index=0)

with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0, value=3000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0.0, value=120.0)
    loan_amount_term = st.number_input("Loan Amount Term (months)", min_value=1.0, value=360.0)
    credit_history = st.selectbox("Credit History (1 = good)", [1.0, 0.0], index=0)

property_area = st.selectbox("Property Area", ["Urban","Semiurban","Rural"], index=0)

if st.button("Predict"):
    # prepare single-row DataFrame with same columns used during training
    def clean_dependents_input(d):
        if isinstance(d, str) and d == "3+":
            return 3
        try:
            return int(d)
        except:
            return np.nan

    total_income = applicant_income + coapplicant_income

    row = {
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Dependents": clean_dependents_input(dependents),
        "Total_Income": total_income,
        "Gender": gender,
        "Married": married,
        "Education": education,
        "Self_Employed": self_employed,
        "Property_Area": property_area
    }

    X = pd.DataFrame([row])

    # predict
    pred_proba = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None
    pred = model.predict(X)[0]

    if pred == 1:
        st.success(f"✅ Loan likely to be APPROVED (probability: {pred_proba:.2f})" if pred_proba is not None else "✅ Loan likely to be APPROVED")
    else:
        st.error(f"❌ Loan likely to be REJECTED (probability: {pred_proba:.2f})" if pred_proba is not None else "❌ Loan likely to be REJECTED")
    
    st.write("**Model explanation**: This result comes from a pipeline that imputes missing values, scales numerics, encodes categoricals and runs a RandomForest classifier.")
