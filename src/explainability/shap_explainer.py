"""
SHAP explainability layer.
- Global: feature importance bar chart + beeswarm summary
- Local: per-customer explanation dict (used by Data Agent via MCP tool)
- Saves SHAP values for every test customer to disk
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path

REPORTS_DIR = Path("models/reports")
MODELS_DIR  = Path("models")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def build_explainer(model, X_train: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, MODELS_DIR / "shap_explainer.pkl")
    print("SHAP explainer saved to models/shap_explainer.pkl")
    return explainer


def compute_shap_values(explainer, X: pd.DataFrame) -> np.ndarray:
    shap_values = explainer.shap_values(X)
    # Handle LightGBM list output if ever switching back
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_df = pd.DataFrame(shap_values, columns=X.columns, index=X.index)
    shap_df.to_csv(MODELS_DIR / "shap_values_test.csv")
    print(f"SHAP values saved: {shap_df.shape} — models/shap_values_test.csv")
    return shap_values


def plot_global_importance(shap_values, X: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    mean_abs = np.abs(shap_values).mean(axis=0)
    feat_imp = pd.Series(mean_abs, index=X.columns).sort_values(ascending=True)
    feat_imp.tail(15).plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title("Global Feature Importance (mean |SHAP value|)")
    ax.set_xlabel("mean |SHAP|")
    path = REPORTS_DIR / "shap_global_importance.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_beeswarm(explainer, X: pd.DataFrame):
    shap_vals_obj = explainer(X)
    shap.plots.beeswarm(shap_vals_obj, max_display=15, show=False)
    path = REPORTS_DIR / "shap_beeswarm.png"
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def explain_customer(
    customer_index: int,
    X_test: pd.DataFrame,
    shap_values: np.ndarray,
    top_n: int = 5,
) -> dict:
    """
    Structured local explanation for one customer.
    Called by the Data Agent via the MCP tool.
    """
    row_shap      = shap_values[customer_index]
    feature_names = X_test.columns.tolist()

    contributions = sorted(
        zip(feature_names, row_shap, X_test.iloc[customer_index].values),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    return {
        "customer_index": customer_index,
        "top_risk_factors": [
            {
                "feature":        feat,
                "shap_value":     round(float(sv), 4),
                "customer_value": round(float(cv), 4),
                "direction":      "increases churn risk" if sv > 0 else "decreases churn risk",
            }
            for feat, sv, cv in contributions[:top_n]
        ],
    }


def run_explainability(model, X_train: pd.DataFrame, X_test: pd.DataFrame):
    print("\n── SHAP explainability ──────────────────────")
    explainer   = build_explainer(model, X_train)
    shap_values = compute_shap_values(explainer, X_test)
    plot_global_importance(shap_values, X_test)
    plot_beeswarm(explainer, X_test)
    print("Explainability complete.")
    return explainer, shap_values