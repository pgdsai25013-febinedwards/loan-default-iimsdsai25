#!/usr/bin/env python3
"""
predict.py

Usage examples:
# Predict from a CSV with multiple rows
python predict.py --input_csv "data/raw/new_customers.csv" --output "predictions.csv"

# Predict a single customer from JSON
python predict.py --json '{"person_age":30,"person_income":50000,"person_home_ownership":"RENT","person_emp_length":5,"loan_intent":"PERSONAL","loan_grade":"C","loan_amnt":15000,"loan_int_rate":12.5,"loan_percent_income":0.3,"cb_person_default_on_file":"N","cb_person_cred_hist_length":3}' 

# Change probability threshold
python predict.py --input_csv new.csv --threshold 0.6
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# --- CONFIG: change if your files are in other locations ---
BASE_DIR = r"C:\Users\evely\OneDrive\IIM\chandrika\Loan Default dataset\loan_project"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

PREPROCESSOR_PATH = os.path.join(DATA_DIR, "preprocessor.pkl")
FEATURES_RAW = [
    # These are the raw columns your pipeline expects (order doesn't matter)
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
    "cb_person_cred_hist_length"
]
XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")  # uses joblib
NN_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")            # optional, kept for reference

# --- Helper functions ---
def load_models():
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}")
    if not os.path.exists(XGBOOST_MODEL_PATH):
        raise FileNotFoundError(f"XGBoost model not found at {XGBOOST_MODEL_PATH}")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    xgb_model = joblib.load(XGBOOST_MODEL_PATH)
    # optionally load NN if you want to predict with it too:
    # nn_model = load_model(NN_MODEL_PATH) if os.path.exists(NN_MODEL_PATH) else None
    return preprocessor, xgb_model

def validate_input_df(df: pd.DataFrame):
    # Ensure required columns exist
    missing = [c for c in FEATURES_RAW if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    # Keep only the columns we need (extra columns are ignored)
    return df[FEATURES_RAW].copy()

def prepare_features(preprocessor, df_raw: pd.DataFrame):
    """
    preprocessor: sklearn ColumnTransformer (fitted)
    df_raw: raw dataframe with columns equal to FEATURES_RAW
    returns: transformed numpy array ready for model.predict_proba
    """
    # NOTE: preprocessor expects the same column names and types used during training.
    try:
        X_transformed = preprocessor.transform(df_raw)
    except Exception as e:
        # Common problem: OneHotEncoder raised error on unseen category.
        raise RuntimeError(
            "Preprocessor.transform failed. Possible causes:\n"
            "- categorical value unseen during training\n"
            "- different column dtypes\n"
            "Original error: " + str(e)
        )
    return X_transformed

def predict_batch(model, X_transformed, threshold=0.5):
    """
    model: sklearn-like estimator with predict_proba (e.g. XGBoost)
    X_transformed: numpy array
    returns DataFrame with columns: prob_default, pred_label
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_transformed)[:, 1]
    else:
        # fallback: some models use predict -> probability-like
        probs = model.predict(X_transformed)
        probs = np.array(probs).reshape(-1)
    preds = (probs >= threshold).astype(int)
    return probs, preds

# --- Main CLI ---
def main():
    parser = argparse.ArgumentParser(description="Predict loan default probability")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_csv", type=str, help="Path to CSV file with raw features (one row per customer)")
    group.add_argument("--json", type=str, help='Single-row JSON string with raw features, e.g. \'{"person_age":30,...}\'')
    parser.add_argument("--output", type=str, help="Path to save predictions as CSV", default=None)
    parser.add_argument("--threshold", type=float, help="Probability threshold for class label (default=0.5)", default=0.5)
    parser.add_argument("--model", type=str, choices=["xgb","nn"], default="xgb", help="Which model to use for prediction (xgb or nn). Default=xgb")
    args = parser.parse_args()

    preprocessor, xgb_model = load_models()
    # optionally load nn if requested
    nn_model = None
    if args.model == "nn":
        if os.path.exists(NN_MODEL_PATH):
            nn_model = load_model(NN_MODEL_PATH)
        else:
            print("Neural network model not found; using XGBoost instead.")
            args.model = "xgb"

    # Build input dataframe
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
    else:
        # parse JSON string
        try:
            data = json.loads(args.json)
        except Exception as e:
            raise ValueError("Invalid JSON string passed to --json") from e
        df = pd.DataFrame([data])

    try:
        df_valid = validate_input_df(df)
    except ValueError as e:
        print("Input validation error:", e)
        sys.exit(1)

    # Prepare features
    try:
        X_ready = prepare_features(preprocessor, df_valid)
    except RuntimeError as e:
        print("Preprocessing error:", e)
        sys.exit(1)

    # Choose model and predict
    if args.model == "xgb":
        probs, preds = predict_batch(xgb_model, X_ready, threshold=args.threshold)
    else:
        # NN route: use keras model.predict
        probs = nn_model.predict(X_ready, verbose=0).reshape(-1)
        preds = (probs >= args.threshold).astype(int)

    # Build results
    results = df_valid.copy().reset_index(drop=True)
    results["prob_default"] = probs
    results["pred_default"] = preds

    # Print results
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\nPredictions:")
    print(results[["prob_default", "pred_default"]].to_string(index=False))

    # Save if requested
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nSaved predictions to {args.output}")

if __name__ == "__main__":
    main()