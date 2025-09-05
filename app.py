# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="centered")

st.title("🏦 Loan Approval Prediction")
st.markdown(
    """
    Enter applicant details to predict loan approval.
    The app expects a serialized scikit-learn pipeline (preprocessor + model) saved as a single file (recommended).
    """
)

MODEL_PATHS = [
    "pipeline.pkl",       # recommended: contains ColumnTransformer + model
    "model.pkl",          # alternative common name
    "best_model.pkl",     # your original version (tuple stored)
    "best_model.pickle",
    "best_model.pkl"      # fallback - matches original file you shared
]

@st.cache_resource
def load_model():
    """Try several common paths and formats, return (pipeline, metadata) or raise."""
    for p in MODEL_PATHS:
        if os.path.exists(p):
            with open(p, "rb") as f:
                obj = pickle.load(f)
            # If you saved a tuple (model, scaler, encoders) like original script, handle that
            if isinstance(obj, tuple) and len(obj) in (2, 3):
                # original structure: (model, scaler, encoders) or similar
                model = obj[0]
                extra = {"scaler": obj[1] if len(obj) > 1 else None, "encoders": obj[2] if len(obj) > 2 else None}
                return model, extra
            # If it's a single pipeline, return it
            return obj, {}
    raise FileNotFoundError(f"No model file found. Checked: {MODEL_PATHS}")

def make_input_df(inputs: dict, feature_order: list = None) -> pd.DataFrame:
    """Create DataFrame from inputs and reorder to feature_order if provided."""
    df = pd.DataFrame([inputs])
    # convert Dependents '3+' to numeric 3 if present
    if "Dependents" in df.columns:
        df["Dependents"] = df["Dependents"].replace({"3+": "3"})
    # ensure numeric types for numeric fields
    for col in ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History"]:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass
    # reorder columns if pipeline expects a specific order
    if feature_order:
        # take intersection to avoid KeyError
        cols = [c for c in feature_order if c in df.columns]
        df = df[cols]
    return df

# ---- Sidebar: sample mode / load info ----
st.sidebar.header("Model / Info")
try:
    model_obj, meta = load_model()
    st.sidebar.success("Model loaded")
    st.sidebar.write("Model type:", type(model_obj).__name__)
    if meta:
        st.sidebar.write("Additional items:", ", ".join([k for k,v in meta.items() if v is not None]))
except Exception as e:
    st.sidebar.error("Model not found")
    st.error(f"Model load error: {e}")
    st.stop()

# If pipeline has attribute 'named_steps' and preprocessor, attempt to extract expected feature names
expected_feature_order = None
try:
    # If it's a pipeline with a ColumnTransformer named 'preprocessor', try to extract numeric+cat names
    if hasattr(model_obj, "named_steps"):
        # If user used a ColumnTransformer and numeric/categorical lists were saved as attributes, we try to fetch them
        # Otherwise, a safe fallback is None (we'll use the input columns as-is)
        # Many pipelines don't expose feature names directly, so we keep this optional.
        if "preprocessor" in model_obj.named_steps:
            pre = model_obj.named_steps["preprocessor"]
            # numeric/onehot extraction is data-dependent; skip a complex attempt to avoid errors
            expected_feature_order = None
except Exception:
    expected_feature_order = None

# ---- User inputs ----
st.header("Applicant details")

col1, col2 = st.columns(2)
with col1:
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
with col2:
    ApplicantIncome = st.number_input("Applicant Income", min_value=0, step=100, value=2500)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0, step=100, value=0)
    LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0.0, step=0.5, value=100.0)
    Loan_Amount_Term = st.selectbox("Loan Amount Term (months)", [12, 36, 60, 120, 180, 240, 300, 360, 480])
    Credit_History = st.selectbox("Credit History", [1.0, 0.0])
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Build the input dictionary (match your training column names)
input_dict = {
    "Gender": Gender,
    "Married": Married,
    "Dependents": Dependents,
    "Education": Education,
    "Self_Employed": Self_Employed,
    "ApplicantIncome": ApplicantIncome,
    "CoapplicantIncome": CoapplicantIncome,
    "LoanAmount": LoanAmount,
    "Loan_Amount_Term": Loan_Amount_Term,
    "Credit_History": Credit_History,
    "Property_Area": Property_Area,
}

st.subheader("Input summary")
st.json(input_dict)

# Prepare dataframe for model
X = make_input_df(input_dict, feature_order=expected_feature_order)

# If model was saved as (model, scaler, encoders) rather than a single pipeline, handle that:
if isinstance(model_obj, tuple) or ("scaler" in meta and meta.get("scaler") is not None):
    # unlikely with our loader, but kept for compatibility with older saving patterns
    st.warning("Model is in legacy format (model, scaler, encoders). Attempting to use it anyway.")
    try:
        # assume model_obj is the classifier if tuple
        if isinstance(model_obj, tuple):
            clf = model_obj[0]
        else:
            clf = model_obj
        scaler = meta.get("scaler")
        encoders = meta.get("encoders", {})
        # apply encoding if encoders provided
        X_proc = X.copy()
        if encoders:
            for col, enc in encoders.items():
                if col in X_proc.columns:
                    X_proc[col] = enc.transform(X_proc[col].astype(str))
        # apply scaler if provided (assume scaler expects numeric matrix)
        if scaler is not None:
            # keep feature order same as scaler used (best-effort)
            X_scaled = scaler.transform(X_proc)
            preds_proba = clf.predict_proba(X_scaled)[0]
            preds = clf.predict(X_scaled)[0]
        else:
            preds_proba = clf.predict_proba(X_proc)[0]
            preds = clf.predict(X_proc)[0]
    except Exception as e:
        st.error(f"Legacy model prediction failed: {e}")
        st.stop()
else:
    # Standard: model_obj is a pipeline that includes preprocessing + classifier
    pipeline = model_obj
    try:
        preds_proba = pipeline.predict_proba(X)[0]
        preds = pipeline.predict(X)[0]
    except Exception as e:
        st.error(f"Prediction failed — check that pipeline expects the same columns. Error: {e}")
        st.stop()

# Map numeric label to human-friendly
label_map = {1: "Approved", 0: "Rejected", "Y": "Approved", "N": "Rejected"}

# Show results
st.header("Prediction")
prob_approved = float(preds_proba[1] if len(preds_proba) > 1 else preds_proba[0])
prob_rejected = float(preds_proba[0] if len(preds_proba) > 1 else 1 - prob_approved)

colp, colq = st.columns([1, 2])
with colp:
    if preds in label_map:
        label = label_map[preds]
    else:
        # if prediction is probability thresholded or 0/1
        try:
            label = "Approved" if int(preds) == 1 else "Rejected"
        except Exception:
            label = str(preds)
    if "Approved" in label:
        st.success(f"✅ {label}")
    else:
        st.error(f"❌ {label}")

with colq:
    st.metric("Confidence (Approved)", f"{prob_approved*100:.1f}%")
    st.progress(int(prob_approved*100))

st.write("---")
st.subheader("Detailed probabilities")
st.write(pd.DataFrame({
    "Class": ["Rejected", "Approved"][:len(preds_proba)],
    "Probability": [prob_rejected, prob_approved][:len(preds_proba)]
}).set_index("Class"))

st.write("")
st.caption("If you change the model file (pipeline vs separate objects), update MODEL_PATHS and loading logic.")

# Optional: let user download example input as CSV
if st.button("Download sample input CSV"):
    csv = X.to_csv(index=False)
    st.download_button("Download CSV", csv, file_name="sample_input.csv", mime="text/csv")
