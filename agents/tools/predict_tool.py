"""
agents/tools/predict_tool.py — XGBoost inference + SHAP explanation.

Used exclusively by Agent 1 (Diagnostic Lead).

Two encoding paths:
  encode_customer(..., feat_35)  → xgb_pipeline.pkl  (SMOTEENN + XGBClassifier)
  encode_customer(..., feat_36)  → shap_explainer.pkl (needs Charge Index)

Why two paths: xgb_pipeline was trained on 35 features (Charge Index excluded
from that pipeline's preprocessing), but shap_explainer was built on the
standalone xgboost_churn.pkl which uses 36 features. We verified this in the
artifact inspection step (xgb_pipeline expects 35, shap_values_test has 36 cols).
"""

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data.loader import BINARY_COLS, NOMINAL_COLS, CONTRACT_ORDINAL

MODELS_DIR = Path("models")


# ── Artifact loading (cached — loaded once per process) ───────────────────────

@lru_cache(maxsize=1)
def _artifacts() -> dict:
    return {
        "pipeline":   joblib.load(MODELS_DIR / "xgb_pipeline.pkl"),
        "xgb_model":  joblib.load(MODELS_DIR / "xgboost_churn.pkl"),
        "shap_exp":   joblib.load(MODELS_DIR / "shap_explainer.pkl"),
        "encoders":   joblib.load(MODELS_DIR / "encoders.pkl"),
        "feat_35":    joblib.load(MODELS_DIR / "feature_names_35.pkl"),
        "feat_36":    joblib.load(MODELS_DIR / "feature_names_36.pkl"),
        "threshold":  joblib.load(MODELS_DIR / "threshold.pkl"),
    }


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_customer(customer_raw: dict, encoders: dict, feature_names: list) -> pd.DataFrame:
    """
    Replicates encode_for_gbm() for a single customer dict.
    Matches the exact transformation chain used during training.

    Args:
        customer_raw:   Dict of raw feature values (pre-encoding strings/numbers)
        encoders:       Loaded from models/encoders.pkl
        feature_names:  Either feat_35 or feat_36 list

    Returns:
        1-row DataFrame with columns in the exact order the model expects
    """
    df = pd.DataFrame([customer_raw])

    # 1. Contract: ordinal mapping (Month-to-month=1, One year=12, Two year=24)
    if "Contract" in df.columns:
        df["Contract"] = (
            df["Contract"].map(encoders.get("Contract", CONTRACT_ORDINAL))
            .fillna(1).astype(int)
        )

    # 2. Binary columns: LabelEncoder (fitted on training data)
    for col in BINARY_COLS:
        if col in df.columns and col in encoders:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))

    # 3. Nominal columns: OneHotEncoder (drop="first", fitted on training data)
    ohe = encoders["_ohe"]
    nominal_present = [c for c in NOMINAL_COLS if c in df.columns]
    if nominal_present:
        ohe_array = ohe.transform(df[nominal_present])
        ohe_cols  = ohe.get_feature_names_out(nominal_present)
        ohe_df    = pd.DataFrame(ohe_array, columns=ohe_cols, index=df.index)
        df = df.drop(columns=nominal_present)
        df = pd.concat([df, ohe_df], axis=1)

    # 4. Pad any missing columns with 0 (handles OHE categories unseen at inference)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    return df[feature_names]


# ── Main prediction function ───────────────────────────────────────────────────

def predict_customer(customer_raw: dict) -> dict:
    """
    Run full inference pipeline on one customer.

    Returns:
        {
            risk_score:   float (0.0–1.0),
            risk_label:   "HIGH" | "LOW",
            threshold:    float,
            shap_drivers: list[dict] — top 5 SHAP contributors
        }
    """
    art = _artifacts()

    # ── Risk score (35-feature pipeline) ─────────────────────────────────────
    X_35         = encode_customer(customer_raw, art["encoders"], art["feat_35"])
    risk_score   = float(art["pipeline"].predict_proba(X_35)[0, 1])
    threshold    = art["threshold"]["best_threshold"]

    # ── SHAP explanation (36-feature standalone model) ────────────────────────
    X_36      = encode_customer(customer_raw, art["encoders"], art["feat_36"])
    shap_vals = art["shap_exp"].shap_values(X_36)

    # Handle both single-array (XGBoost) and list (LightGBM) outputs
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    row_shap      = shap_vals[0]
    feature_names = X_36.columns.tolist()
    raw_values    = X_36.iloc[0].values

    # Sort features by |SHAP| descending, take top 5
    contributions = sorted(
        zip(feature_names, row_shap, raw_values),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    shap_drivers = [
        {
            "feature":    feat,
            "shap_value": round(float(sv), 4),
            "raw_value":  round(float(rv), 4),
            "direction":  "increases churn risk" if sv > 0 else "decreases churn risk",
        }
        for feat, sv, rv in contributions[:5]
    ]

    return {
        "risk_score":   round(risk_score, 4),
        "risk_label":   "HIGH" if risk_score >= threshold else "LOW",
        "threshold":    round(threshold, 4),
        "shap_drivers": shap_drivers,
    }