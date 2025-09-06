# app.py
import streamlit as st
import pickle
import pandas as pd

with open('best_model.pkl','rb') as f:
    model = pickle.load(f)

st.title("Loan Approval Prediction")

# Build UI for inputs (put actual fields)
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.number_input("Loan Amount Term (days)", min_value=0)
Credit_History = st.selectbox("Credit History", [1.0, 0.0])

Gender = st.selectbox("Gender", ['Male','Female'])
Married = st.selectbox("Married", ['Yes','No'])
Education = st.selectbox("Education", ['Graduate','Not Graduate'])
Self_Employed = st.selectbox("Self Employed", ['Yes','No'])
Property_Area = st.selectbox("Property Area", ['Urban','Semiurban','Rural'])

input_dict = {
    'ApplicantIncome':[ApplicantIncome],
    'CoapplicantIncome':[CoapplicantIncome],
    'LoanAmount':[LoanAmount],
    'Loan_Amount_Term':[Loan_Amount_Term],
    'Credit_History':[Credit_History],
    'Gender':[Gender],
    'Married':[Married],
    'Education':[Education],
    'Self_Employed':[Self_Employed],
    'Property_Area':[Property_Area]
}

X_new = pd.DataFrame.from_dict(input_dict)
if st.button("Predict"):
    pred = model.predict(X_new)[0]
    st.success("Approved" if pred==1 else "Rejected")
