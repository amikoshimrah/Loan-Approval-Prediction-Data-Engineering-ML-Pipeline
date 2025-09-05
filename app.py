# app.py
import streamlit as st
import pandas as pd
import pickle

# Load the trained model (could be model only, or model + scaler, or model+scaler+encoders)
with open("best_model.pkl", "rb") as f:
    loaded = pickle.load(f)

# Handle different formats
model, scaler, encoders = None, None, {}

if isinstance(loaded, tuple):
    if len(loaded) == 3:
        model, scaler, encoders = loaded
    elif len(loaded) == 2:
        model, scaler = loaded
    elif len(loaded) == 1:
        model = loaded[0]
else:
    model = loaded

st.title("🏦 Loan Approval Prediction App")

st.write("Fill in the applicant details below to check loan approval:")

# --- Input fields ---
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0, step=100)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=100)
loan_amount = st.number_input("Loan Amount", min_value=0, step=10)
loan_amount_term = st.selectbox("Loan Amount Term (months)", [12, 36, 60, 120, 180, 240, 300, 360, 480])
credit_history = st.selectbox("Credit History", [0, 1])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# --- Prepare DataFrame ---
input_dict = {
    "Gender": [gender],
    "Married": [married],
    "Dependents": [dependents],
    "Education": [education],
    "Self_Employed": [self_employed],
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_amount_term],
    "Credit_History": [credit_history],
    "Property_Area": [property_area],
}
df_input = pd.DataFrame(input_dict)

# Apply encoders if available
if encoders:
    for col in df_input.columns:
        if col in encoders:
            df_input[col] = encoders[col].transform(df_input[col].astype(str))

# Scale numeric if scaler exists
if scaler:
    df_input = scaler.transform(df_input)

# --- Predict ---
if st.button("Predict Loan Approval"):
    prediction = model.predict(df_input)[0]
    if prediction == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Rejected")
