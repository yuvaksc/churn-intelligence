"""
api/dependencies.py — Shared application state loaded once at startup.

initialize_state() is called in FastAPI's lifespan event.
get_state() is used as a FastAPI dependency in every route that needs data.

What gets loaded once:
  - Full test set (eng_test_raw, X_test, y_test) — reconstructed deterministically
  - Risk scores for all 1409 test customers (pre-computed, avoids per-request scoring)
  - SHAP values from shap_values_test.csv (pre-computed by run_pipeline.py)
  - Threshold from threshold.pkl
  - Feature names (feat_35) for column ordering
"""

import sys
import asyncio
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import (
    load_raw, clean, engineer_features,
    encode_for_gbm, three_way_split,
)

MODELS_DIR = Path("models")


@dataclass
class AppState:
    eng_test_raw: pd.DataFrame   # pre-encoding test rows — passed to war room agents
    X_test:       pd.DataFrame   # encoded test rows — used for pre-scoring
    y_test:       pd.Series      # ground truth labels
    risk_scores:  np.ndarray     # pre-computed predict_proba for all test customers
    shap_df:      pd.DataFrame   # SHAP values (1409 × 36) from shap_values_test.csv
    threshold:    float          # from threshold.pkl
    feat_35:      list[str]      # column order for xgb_pipeline.pkl


_state: AppState | None = None


def initialize_state() -> AppState:
    """
    Load and cache all expensive resources.
    Called once in FastAPI lifespan — subsequent calls return cached state.
    """
    global _state
    if _state is not None:
        return _state

    print("  Loading test set (deterministic split)...")
    raw_df    = load_raw()
    eng_df    = engineer_features(clean(raw_df))
    gbm_df, _ = encode_for_gbm(eng_df)
    _, _, X_test, _, _, y_test = three_way_split(gbm_df)
    eng_test  = eng_df.loc[X_test.index]

    print("  Pre-scoring all test customers...")
    pipeline  = joblib.load(MODELS_DIR / "xgb_pipeline.pkl")
    feat_35   = joblib.load(MODELS_DIR / "feature_names_35.pkl")
    threshold = joblib.load(MODELS_DIR / "threshold.pkl")["best_threshold"]
    scores    = pipeline.predict_proba(X_test[feat_35])[:, 1]

    print("  Loading SHAP values...")
    shap_df = pd.read_csv(MODELS_DIR / "shap_values_test.csv", index_col=0)

    _state = AppState(
        eng_test_raw=eng_test,
        X_test=X_test,
        y_test=y_test,
        risk_scores=scores,
        shap_df=shap_df,
        threshold=threshold,
        feat_35=feat_35,
    )

    high_risk = int((scores >= threshold).sum())
    print(f"  Ready — {len(eng_test)} customers, {high_risk} above threshold ({threshold:.0%})")
    return _state


def get_state() -> AppState:
    """FastAPI dependency — inject into route handlers."""
    if _state is None:
        raise RuntimeError("App state not initialised. Is the lifespan running?")
    return _state


def get_customer_raw(customer_id: int, state: AppState) -> dict | None:
    """
    Look up a test customer by pandas index.
    Returns the pre-encoding feature dict, or None if not found.
    """
    if customer_id not in state.eng_test_raw.index:
        return None
    row = state.eng_test_raw.loc[customer_id]
    result = {}
    for col, val in row.items():
        if isinstance(val, (np.integer,)):
            result[col] = int(val)
        elif isinstance(val, (np.floating,)):
            result[col] = float(val)
        else:
            result[col] = val
    return result


def get_top_shap_drivers(customer_id: int, state: AppState, top_n: int = 5) -> list[dict]:
    """
    Pull pre-computed SHAP values for one customer from shap_values_test.csv.
    Returns top_n drivers sorted by |SHAP| descending.
    """
    if customer_id not in state.shap_df.index:
        return []

    row   = state.shap_df.loc[customer_id]
    items = [(feat, float(val)) for feat, val in row.items()]
    items.sort(key=lambda x: abs(x[1]), reverse=True)

    return [
        {
            "feature":    feat,
            "shap_value": round(sv, 4),
            "raw_value":  0.0,   # raw value is in the encoded space — less useful here
            "direction":  "increases churn risk" if sv > 0 else "decreases churn risk",
        }
        for feat, sv in items[:top_n]
    ]