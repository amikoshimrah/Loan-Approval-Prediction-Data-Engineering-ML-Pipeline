# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from typing import Any, Dict

st.set_page_config(page_title="Loan Approval", layout="centered")
st.title("🏦 Loan Approval Prediction (robust loader)")

MODEL_PATH = "best_model.pkl"

# -------------------------
# Utilities: robust loader
# -------------------------
def load_artifact(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found in working directory.")
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    # Normalize into a dict with keys: model, features, scaler, encoders
    artifact = {"model": None, "features": None, "scaler": None, "encoders": None}
    if isinstance(loaded, dict):
        artifact["model"] = loaded.get("model") or loaded.get("pipeline") or loaded.get("estimator")
        artifact["features"] = loaded.get("features") or loaded.get("feature_names") or loaded.get("columns")
        artifact["scaler"] = loaded.get("scaler")
        artifact["encoders"] = loaded.get("encoders") or loaded.get("label_encoders")
    elif isinstance(loaded, (list, tuple)):
        # common conventions: (model, scaler, encoders) or (model, scaler)
        if len(loaded) == 3:
            artifact["model"], artifact["scaler"], artifact["encoders"] = loaded
        elif len(loaded) == 2:
            artifact["model"], artifact["scaler"] = loaded
        elif len(loaded) == 1:
            artifact["model"] = loaded[0]
    else:
        # loaded is likely a single estimator/pipeline
        artifact["model"] = loaded
    return artifact

def infer_features(artifact: Dict[str, Any]):
    # priority: artifact['features'] -> scaler.feature_names_in_ -> model.feature_names_in_ ->
    # ColumnTransformer inside pipeline: preprocessor.feature_names_in_
    if artifact.get("features"):
        return list(artifact["features"])
    scaler = artifact.get("scaler")
    model = artifact.get("model")
    # scaler
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        try:
            return list(getattr(scaler, "feature_names_in_"))
        except Exception:
            pass
    # model top-level
    if model is not None and hasattr(model, "feature_names_in_"):
        try:
            return list(getattr(model, "feature_names_in_"))
        except Exception:
            pass
    # pipeline with preprocessor
    try:
        if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
            pre = model.named_steps["preprocessor"]
            if hasattr(pre, "feature_names_in_"):
                return list(getattr(pre, "feature_names_in_"))
            # ColumnTransformer sometimes doesn't expose feature_names_in_; try model.feature_names_in_ again below
    except Exception:
        pass
    # fallback to commonly used German credit fields (same as original UI)
    fallback = [
        "Gender", "Married", "Dependents", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
        "Credit_History", "Property_Area"
    ]
    return fallback

def ensure_columns(df: pd.DataFrame, required_cols):
    df2 = df.copy()
    missing = [c for c in required_cols if c not in df2.columns]
    if missing:
        # add missing with NaN
        for c in missing:
            df2[c] = np.nan
    # reorder columns to required_cols
    df2 = df2.loc[:, required_cols]
    return df2, missing

def apply_encoders(encoders: Dict[str, Any], X: pd.DataFrame):
    X2 = X.copy()
    if not encoders:
        return X2
    for col, enc in encoders.items():
        if col not in X2.columns:
            continue
        try:
            # handle LabelEncoder-like and sklearn encoders with transform
            X2[col] = enc.transform(X2[col].astype(str))
        except Exception:
            # fallback: try map via classes_ if present
            if hasattr(enc, "classes_"):
                mapping = {v: i for i, v in enumerate(getattr(enc, "classes_"))}
                X2[col] = X2[col].map(mapping).fillna(-1)
            else:
                # leave as-is
                pass
    return X2

# -------------------------
# Load artifact
# -------------------------
try:
    artifact = load_artifact(MODEL_PATH)
except Exception as e:
    st.error(str(e))
    st.stop()

model = artifact.get("model")
scaler = artifact.get("scaler")
encoders = artifact.get("encoders") or {}
expected_features = infer_features(artifact)

st.sidebar.header("Model info")
if model is None:
    st.sidebar.error("Model not found inside the pickle.")
else:
    st.sidebar.success(f"Loaded model: {type(model).__name__}")
    if hasattr(model, "named_steps"):
        st.sidebar.info("Detected pipeline with named_steps (may contain preprocessing).")
if scaler is not None and not hasattr(model, "named_steps"):
    st.sidebar.info(f"Scaler present: {type(scaler).__name__}")
if encoders:
    st.sidebar.info(f"Encoders: {', '.join(list(encoders.keys()))}")
st.sidebar.write("Expected features (first 20):")
st.sidebar.write(expected_features[:20])

# -------------------------
# Build input UI (dynamic from expected_features where sensible)
# -------------------------
st.markdown("### Applicant details (fill fields below)")

# Provide sensible defaults if expected features match common names
defaults = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 2500.0,
    "CoapplicantIncome": 0.0,
    "LoanAmount": 100.0,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Urban"
}

# Build simple widgets for the features that we know; otherwise text inputs
input_vals = {}
for col in expected_features:
    if col == "Gender":
        input_vals[col] = st.selectbox("Gender", ["Male", "Female"], index=0 if defaults["Gender"] == "Male" else 1)
    elif col == "Married":
        input_vals[col] = st.selectbox("Married", ["Yes", "No"], index=0 if defaults["Married"] == "Yes" else 1)
    elif col == "Dependents":
        input_vals[col] = st.selectbox("Dependents", ["0", "1", "2", "3+"], index=0)
    elif col == "Education":
        input_vals[col] = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0)
    elif col == "Self_Employed":
        input_vals[col] = st.selectbox("Self Employed", ["Yes", "No"], index=1)
    elif col == "Property_Area":
        input_vals[col] = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], index=0)
    elif col == "Credit_History":
        input_vals[col] = st.selectbox("Credit History", [0, 1], index=1)
    elif col == "Loan_Amount_Term":
        input_vals[col] = st.selectbox("Loan Amount Term (months)", [12, 36, 60, 120, 180, 240, 300, 360, 480], index=7)
    else:
        # numeric-like names get numeric input, otherwise text
        if any(k in col.lower() for k in ("income", "loan", "amount", "term", "credit", "rate", "age", "score")):
            default = defaults.get(col, 0.0)
            step = 100.0 if "income" in col.lower() else (10.0 if "loan" in col.lower() or "amount" in col.lower() else 1.0)
            input_vals[col] = st.number_input(col, min_value=0.0, value=float(default), step=step)
        else:
            input_vals[col] = st.text_input(col, value=str(defaults.get(col, "")))

# Allow user to override or add extra columns manually if model expects more
if st.checkbox("Show advanced: add/override input fields"):
    add_col = st.text_input("Add/override column name (leave blank to skip)")
    if add_col:
        val = st.text_input(f"Value for {add_col}", value="")
        input_vals[add_col] = val
        if add_col not in expected_features:
            expected_features.append(add_col)
            st.info(f"Added {add_col} to expected features for this session.")

# -------------------------
# Predict button logic
# -------------------------
if st.button("Predict"):
    # Build single-row DataFrame from input_vals
    df_input = pd.DataFrame([input_vals])

    # Ensure all expected columns present and in right order
    X_for_model, missing_cols = ensure_columns(df_input, expected_features)

    if missing_cols:
        st.warning(f"Added missing columns with NaN: {missing_cols}")

    st.write("Input (after aligning with model features):")
    st.dataframe(X_for_model.T)

    # Decide path: pipeline (preprocessor inside) or separate scaler/encoders
    is_pipeline = hasattr(model, "named_steps") if model is not None else False

    try:
        if is_pipeline:
            # If the pipeline contains ColumnTransformer, it will check columns internally
            pred = model.predict(X_for_model)
            proba = None
            try:
                proba_arr = model.predict_proba(X_for_model)
                if proba_arr.shape[1] == 2:
                    proba = float(proba_arr[:, 1][0])
                else:
                    proba = float(np.max(proba_arr))
            except Exception:
                proba = None
        else:
            # non-pipeline: apply encoders and scaler if present, then call model.predict
            X_proc = X_for_model.copy()

            # Apply encoders first (if present)
            if encoders:
                X_proc = apply_encoders(encoders, X_proc)

            # Try to drop any columns that model does not expect (if model has feature_names_in_)
            if hasattr(model, "feature_names_in_"):
                model_features = list(getattr(model, "feature_names_in_"))
                # ensure order
                X_proc, _ = ensure_columns(X_proc, model_features)
            # Apply scaler if present
            if scaler is not None:
                try:
                    X_scaled = scaler.transform(X_proc)
                except Exception as e:
                    # Try numeric conversion then retry
                    for c in X_proc.columns:
                        X_proc[c] = pd.to_numeric(X_proc[c], errors="coerce").fillna(0)
                    X_scaled = scaler.transform(X_proc)
                pred = model.predict(X_scaled)
                try:
                    proba_arr = model.predict_proba(X_scaled)
                    if proba_arr.shape[1] == 2:
                        proba = float(proba_arr[:, 1][0])
                    else:
                        proba = float(np.max(proba_arr))
                except Exception:
                    proba = None
            else:
                # No scaler, pass dataframe/array directly (try model.predict on df)
                try:
                    pred = model.predict(X_proc)
                    try:
                        proba_arr = model.predict_proba(X_proc)
                        if proba_arr.shape[1] == 2:
                            proba = float(proba_arr[:, 1][0])
                        else:
                            proba = float(np.max(proba_arr))
                    except Exception:
                        proba = None
                except Exception as e:
                    # final fallback: convert to numeric array
                    X_num = X_proc.apply(pd.to_numeric, errors="coerce").fillna(0).values
                    pred = model.predict(X_num)
                    try:
                        proba_arr = model.predict_proba(X_num)
                        if proba_arr.shape[1] == 2:
                            proba = float(proba_arr[:, 1][0])
                        else:
                            proba = float(np.max(proba_arr))
                    except Exception:
                        proba = None

            # If predict returned array
            if isinstance(pred, (list, np.ndarray)):
                pred = pred[0]
            else:
                # try cast
                try:
                    pred = int(pred)
                except Exception:
                    pass

        # Show result
        if str(pred) in ("1", "Yes", "yes", "True", "true", 1):
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        if 'proba' in locals() and proba is not None:
            st.info(f"Predicted positive probability: {proba:.4f}")

    except Exception as e:
        # If ColumnTransformer raised missing columns, try to extract message and show helpful hint
        msg = str(e)
        st.error("Prediction failed.")
        st.exception(e)
        # Common guidance
        st.markdown(
            """
            **Guidance**
            - The saved model expects a specific set of input columns (names & order).
            - This app filled missing columns with `NaN` automatically; ColumnTransformer may still reject `NaN` for some columns.
            - To fix:
              1. Ensure your `best_model.pkl` contains feature names under key `"features"` (a list), or the pipeline was saved with `feature_names_in_`.
              2. Re-save the artifact as: `pickle.dump({'model': pipeline, 'features': X.columns.tolist()}, f)`.
              3. If using separate encoders/scaler, re-save as `{'model': model, 'scaler': scaler, 'encoders': encoders, 'features': X.columns.tolist()}`.
            - Missing / expected columns (first 20): see sidebar.
            """
        )
