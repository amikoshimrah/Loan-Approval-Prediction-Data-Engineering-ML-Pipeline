import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
from io import BytesIO

st.set_page_config(page_title="🏦 Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Prediction App")
st.write("Upload applicant details (single or batch) to predict loan approval.")

# --------------------
# Load Model
# --------------------
MODEL_FILE = "loan_approval_model.pkl"

def load_model(path):
    try:
        return joblib.load(path)
    except:
        with open(path, "rb") as f:
            return pickle.load(f)

if not os.path.exists(MODEL_FILE):
    st.error(f"Model file `{MODEL_FILE}` not found. Please place it in the app directory.")
    st.stop()

model_obj = load_model(MODEL_FILE)

# Detect format
model, scaler, encoders = None, None, None
if isinstance(model_obj, tuple) and len(model_obj) == 3:
    model, scaler, encoders = model_obj
    model_type = "tuple"
else:
    model = model_obj
    model_type = "pipeline"

st.sidebar.info(f"Loaded model type: **{model_type}**")

# --------------------
# Feature Inputs
# --------------------
categorical_features = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area"
]
numeric_features = [
    "ApplicantIncome", "CoapplicantIncome",
    "LoanAmount", "Loan_Amount_Term", "Credit_History"
]

st.header("Single Applicant Prediction")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_emp = st.selectbox("Self Employed", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0, step=100)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=100)
    loan_amount = st.number_input("Loan Amount", min_value=0, step=10)
    loan_term = st.selectbox("Loan Amount Term", [12, 36, 60, 120, 180, 240, 300, 360, 480])
    credit_history = st.selectbox("Credit History", [0, 1])

input_dict = {
    "Gender": [gender],
    "Married": [married],
    "Dependents": [dependents],
    "Education": [education],
    "Self_Employed": [self_emp],
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_term],
    "Credit_History": [credit_history],
    "Property_Area": [property_area],
}
df_input = pd.DataFrame(input_dict)

# Apply preprocessing for tuple format
def preprocess(df):
    df_copy = df.copy()
    if encoders:
        for col, enc in encoders.items():
            if col in df_copy:
                df_copy[col] = enc.transform(df_copy[col].astype(str))
    if scaler:
        df_copy[numeric_features] = scaler.transform(df_copy[numeric_features])
    return df_copy

# Predict button
if st.button("Predict Loan Approval"):
    try:
        if model_type == "tuple":
            X = preprocess(df_input)
            pred = model.predict(X)[0]
        else:
            pred = model.predict(df_input)[0]
        if pred == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
