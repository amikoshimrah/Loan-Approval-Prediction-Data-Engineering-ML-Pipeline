import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import os
from io import BytesIO

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Predictor")
st.write("Load your trained model (`loan_approval_model.pkl`) and predict single or batch loan approvals.")

# ---------------------------
# Expected features (from notebook)
# ---------------------------
# Numeric & categorical features used in the notebook pipeline
NUMERIC_FEATURES = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"
]
CATEGORICAL_FEATURES = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

st.sidebar.header("Model")
DEFAULT_MODEL = "loan_approval_model.pkl"

def try_load_model_path(path):
    """Try to load with joblib first (pipeline), fallback to pickle."""
    try:
        m = joblib.load(path)
        return m, None
    except Exception as e1:
        try:
            with open(path, "rb") as f:
                m = pickle.load(f)
            return m, None
        except Exception as e2:
            return None, f"joblib error: {e1}\npickle error: {e2}"

model = None
load_err = None

# Auto-load default if exists
if os.path.exists(DEFAULT_MODEL):
    model, load_err = try_load_model_path(DEFAULT_MODEL)
    if model:
        st.sidebar.success(f"Auto-loaded model: {DEFAULT_MODEL}")
    else:
        st.sidebar.error(f"Found {DEFAULT_MODEL} but failed to load. See details in app logs.")

# Allow uploading a model file (pkl/joblib)
uploaded_model = st.sidebar.file_uploader("Or upload a model file (.pkl / .joblib)", type=["pkl", "joblib"])
if uploaded_model is not None:
    # joblib.load can accept file-like only in some versions; fallback to pickle.load
    try:
        uploaded_model.seek(0)
        try:
            model = joblib.load(uploaded_model)
        except Exception:
            uploaded_model.seek(0)
            model = pickle.load(uploaded_model)
        st.sidebar.success("Uploaded model loaded.")
        load_err = None
    except Exception as e:
        model = None
        load_err = str(e)
        st.sidebar.error(f"Failed to load uploaded model: {load_err}")

if model is None:
    st.info("No model loaded. Place `loan_approval_model.pkl` in the app folder or upload one in the sidebar.")
    st.stop()

# ---------------------------
# Normalize model shapes
# ---------------------------
# Your original app stored (model, scaler, encoders) in a pickle; other approach is to store a full Pipeline.
# We'll detect these cases and prepare a predict wrapper.
def make_predictor(m):
    """
    Returns a function predict(df) -> (preds, proba_df_or_none)
    Accepts a pandas DataFrame containing the expected columns (strings).
    """
    # Case A: full sklearn Pipeline or estimator supporting predict / predict_proba
    if hasattr(m, "predict"):
        def predict_fn(df):
            preds = m.predict(df)
            proba = None
            if hasattr(m, "predict_proba"):
                try:
                    proba_arr = m.predict_proba(df)
                    proba = pd.DataFrame(proba_arr,
                                         columns=[f"prob_class_{i}" for i in range(proba_arr.shape[1])])
                except Exception:
                    proba = None
            return preds, proba
        return predict_fn, "pipeline"

    # Case B: older tuple saved as (model, scaler, encoders)
    if isinstance(m, (list, tuple)) and len(m) >= 1:
        # try to unpack
        try:
            if len(m) == 3:
                model_obj, scaler, encoders = m
            elif len(m) == 2:
                model_obj, scaler = m
                encoders = {}
            else:
                # not a recognized tuple; fall back to raising
                raise ValueError("Unrecognized pickle structure")
        except Exception as e:
            raise RuntimeError(f"Could not interpret loaded object: {e}")

        def predict_fn(df):
            # copy to avoid modifying original
            X = df.copy()
            # apply encoders (if available) to categorical columns
            if encoders:
                for col, enc in encoders.items():
                    if col in X.columns:
                        try:
                            X[col] = enc.transform(X[col].astype(str))
                        except Exception:
                            # attempt to map unseen values to -1 if label encoder style fails
                            try:
                                X[col] = X[col].astype(str).map(lambda v: enc.transform([v])[0])
                            except Exception:
                                # leave as is
                                pass
            # scale numeric columns if scaler provided
            if scaler is not None:
                # ensure numeric order matches NUMERIC_FEATURES
                num_df = X[NUMERIC_FEATURES].astype(float)
                try:
                    scaled = scaler.transform(num_df)
                    scaled_df = pd.DataFrame(scaled, columns=NUMERIC_FEATURES, index=X.index)
                    # replace numeric columns
                    for c in NUMERIC_FEATURES:
                        X[c] = scaled_df[c]
                except Exception:
                    # if scaling fails, continue without scaling
                    pass
            preds = model_obj.predict(X)
            proba = None
            if hasattr(model_obj, "predict_proba"):
                try:
                    proba_arr = model_obj.predict_proba(X)
                    proba = pd.DataFrame(proba_arr,
                                         columns=[f"prob_class_{i}" for i in range(proba_arr.shape[1])],
                                         index=X.index)
                except Exception:
                    proba = None
            return preds, proba

        return predict_fn, "tuple"

    raise RuntimeError("Loaded object is not a recognisable estimator or tuple of (model, scaler, encoders).")

try:
    predictor, model_form = make_predictor(model)
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

st.sidebar.write(f"Model form detected: **{model_form}**")
try:
    st.sidebar.write(f"Model class: {model.__class__.__name__}")
except Exception:
    pass

# ---------------------------
# Single-row prediction UI
# ---------------------------
st.header("Single prediction")
st.write("Fill in features (fields match the notebook's preprocessing columns).")

# Categorical widgets
col1, col2 = st.columns([1,1])
with col1:
    gender = st.selectbox("Gender", options=["Male", "Female"])
    married = st.selectbox("Married", options=["Yes", "No"])
    dependents = st.selectbox("Dependents", options=["0","1","2","3+"])
    education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
with col2:
    self_employed = st.selectbox("Self Employed", options=["Yes", "No"])
    property_area = st.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"])
    credit_history = st.selectbox("Credit History", options=[0,1])

# Numeric widgets
applicant_income = st.number_input("Applicant Income", min_value=0.0, value=2500.0, step=100.0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0, step=100.0)
loan_amount = st.number_input("Loan Amount (approx.)", min_value=0.0, value=100.0, step=10.0)
loan_amount_term = st.number_input("Loan Amount Term (months)", min_value=1.0, value=360.0, step=12.0)

# Build DataFrame in the expected column order
single_dict = {
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_amount_term,
    "Credit_History": credit_history,
    "Property_Area": property_area,
}
X_single = pd.DataFrame([single_dict], columns=ALL_FEATURES)

if st.button("Predict single row"):
    try:
        preds, proba_df = predictor(X_single)
        st.subheader("Result")
        st.write("Predicted label:", preds[0])
        if proba_df is not None:
            # if binary, show positive class probability if available
            if proba_df.shape[1] == 2:
                st.write(f"Predicted probability (positive class): {proba_df.iloc[0,1]:.4f}")
            else:
                st.write("Probabilities:")
                st.dataframe(proba_df.T)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")

# ---------------------------
# Batch prediction
# ---------------------------
st.header("Batch prediction (CSV / Excel)")
st.write("Upload a CSV / Excel with the same columns as the training features (column names must match).")

uploaded_df = st.file_uploader("Upload data for batch prediction", type=["csv","xlsx","xls"], key="batch")
if uploaded_df is not None:
    try:
        if str(uploaded_df.name).lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_df)
        else:
            df = pd.read_csv(uploaded_df)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        df = None

    if df is not None:
        st.write("Preview:")
        st.dataframe(df.head(5))

        # quick column check
        missing = [c for c in ALL_FEATURES if c not in df.columns]
        if missing:
            st.warning(f"Uploaded file is missing expected columns (sample): {missing[:6]}")

        if st.button("Run batch predictions"):
            try:
                preds, proba_df = predictor(df)
                out = df.copy()
                out["prediction"] = preds
                if proba_df is not None:
                    # proba_df may be numpy or DataFrame — normalize
                    if isinstance(proba_df, pd.DataFrame):
                        out = out.join(proba_df.reset_index(drop=True))
                    else:
                        # assume numpy array
                        try:
                            par = pd.DataFrame(proba_df, columns=[f"prob_class_{i}" for i in range(proba_df.shape[1])])
                            out = out.reset_index(drop=True).join(par)
                        except Exception:
                            pass

                st.success("Predictions done — sample:")
                st.dataframe(out.head(10))

                # prepare download
                towrite = BytesIO()
                out.to_csv(towrite, index=False)
                towrite.seek(0)
                st.download_button("Download predictions CSV", data=towrite, file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

# ---------------------------
# Helpful notes / diagnostics
# ---------------------------
st.sidebar.header("Notes & Diagnostics")
st.sidebar.write("""
- Preferred: save a **single scikit-learn Pipeline** that includes preprocessing (ColumnTransformer) + estimator into `loan_approval_model.pkl`.
- Backwards-compatible: this app also supports older pickles saved as `(model, scaler, encoders)` (your earlier app did this) :contentReference[oaicite:4]{index=4}.
- The notebook used these feature lists and a ColumnTransformer/one-hot approach — the UI fields map to those columns: numeric + categorical features from the notebook :contentReference[oaicite:5]{index=5}.
- If your pickle uses custom transformer classes, make sure their code is importable by Streamlit (place the `.py` in the same folder).
""")

