import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Path to the pickle
MODEL_PKL = "loan_approval_model.pkl"

# --- Load saved object (robust) ---
with open(MODEL_PKL, "rb") as f:
    saved = pickle.load(f)

# Initialize placeholders
model = None
scaler = None
encoders = None
pipeline = None

# Interpret saved object
if isinstance(saved, dict):
    # common keys: 'model','pipeline','scaler','encoders'
    pipeline = saved.get("pipeline") or saved.get("model") if isinstance(saved.get("model"), type(saved.get("model"))) else saved.get("pipeline")
    model = saved.get("model") if saved.get("model") is not None else saved.get("estimator")
    scaler = saved.get("scaler")
    encoders = saved.get("encoders")
    # if pipeline stored under 'pipeline' prefer that
    if saved.get("pipeline") is not None:
        pipeline = saved.get("pipeline")
elif isinstance(saved, (list, tuple)):
    # try to guess contents by length and type
    if len(saved) == 3:
        model, scaler, encoders = saved
    elif len(saved) == 2:
        # ambiguous: could be (model, encoders) or (model, scaler)
        # detect by type: encoders often dict, scaler is usually a scaler object (has .transform)
        a, b = saved
        if isinstance(b, dict):
            model, encoders = a, b
        else:
            # assume scaler-like if it has transform()
            if hasattr(b, "transform"):
                model, scaler = a, b
            else:
                model, encoders = a, b
    elif len(saved) == 1:
        model = saved[0]
    else:
        # fallback: take first as model
        model = saved[0]
else:
    # single object saved — likely a pipeline or estimator
    model = saved

# If model itself is a Pipeline (sklearn pipeline), use that as pipeline
from sklearn.pipeline import Pipeline
if isinstance(model, Pipeline):
    pipeline = model
    model = None

if isinstance(pipeline, Pipeline):
    # prefer to call pipeline.predict on raw inputs — no manual scaling/encoding needed
    MODEL_TYPE = "pipeline"
elif model is not None:
    MODEL_TYPE = "separate"
else:
    MODEL_TYPE = "unknown"

# Sanity print (helpful when running locally)
print("Loaded model type:", MODEL_TYPE)
print("pipeline:", type(pipeline))
print("model:", type(model))
print("scaler:", type(scaler))
print("encoders:", type(encoders))
