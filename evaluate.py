import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "src")

from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)
from data.loader import (
    load_raw, clean, engineer_features,
    encode_for_gbm, three_way_split,
)
from agents.tools.predict_tool import encode_customer

MODELS_DIR  = Path("models")
REPORTS_DIR = Path("models/reports")
REPORTS_DIR.mkdir(exist_ok=True)


def load_test_split():
    raw_df   = load_raw()
    clean_df = clean(raw_df)
    eng_df   = engineer_features(clean_df)
    gbm_df, _ = encode_for_gbm(eng_df)
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(gbm_df)
    eng_test = eng_df.loc[X_test.index]
    return eng_test, X_test, y_test


def score_all_customers(X_test: pd.DataFrame) -> np.ndarray:
    """
    Fast batch scoring — calls pipeline.predict_proba directly on all 1409 rows.
    Bypasses SHAP (not needed for metrics) so this runs in ~1 second total.
    """
    pipeline = joblib.load(MODELS_DIR / "xgb_pipeline.pkl")
    feat_35  = joblib.load(MODELS_DIR / "feature_names_35.pkl")
    return pipeline.predict_proba(X_test[feat_35])[:, 1]


def compute_metrics(y_true, y_prob, threshold, monthly_charges):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    f1        = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    roc_auc   = roc_auc_score(y_true, y_prob)
    avg_prec  = average_precision_score(y_true, y_prob)

    # Business metrics
    # True positives: actual churners we correctly flag and intervene on
    tp_mask  = (y_pred == 1) & (y_true == 1)
    fp_mask  = (y_pred == 1) & (y_true == 0)
    fn_mask  = (y_pred == 0) & (y_true == 1)

    revenue_caught  = monthly_charges[tp_mask].sum()    # monthly revenue saved if retained
    revenue_missed  = monthly_charges[fn_mask].sum()    # churners we failed to flag
    false_pos_spend = monthly_charges[fp_mask].sum()    # charges of customers who weren't going to leave

    interventions_total = int(y_pred.sum())
    actual_churners     = int(y_true.sum())

    return {
        "threshold":            round(threshold, 4),
        "f1":                   round(f1, 4),
        "precision":            round(precision, 4),
        "recall":               round(recall, 4),
        "roc_auc":              round(roc_auc, 4),
        "avg_precision":        round(avg_prec, 4),
        "true_positives":       int(tp),
        "false_positives":      int(fp),
        "false_negatives":      int(fn),
        "true_negatives":       int(tn),
        "interventions_fired":  interventions_total,
        "actual_churners":      actual_churners,
        "revenue_at_risk_caught_monthly":  round(float(revenue_caught), 2),
        "revenue_missed_monthly":          round(float(revenue_missed), 2),
        "false_positive_spend_monthly":    round(float(false_pos_spend), 2),
    }


def load_ml_baseline() -> dict | None:
    """Load metrics from run_pipeline.py output if it exists."""
    path = REPORTS_DIR / "metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def print_report(m: dict, baseline: dict | None):
    sep = "═" * 56

    print(f"\n{sep}")
    print("  AGENT ROUTING EVALUATION  —  Test Set (n=1409)")
    print(sep)

    print(f"\n  {'Metric':<32} {'Agent':>10}  {'ML Baseline':>12}")
    print("  " + "─" * 52)

    def row(label, key, fmt=".4f"):
        agent_val = m[key]
        base_val  = baseline.get(key) if baseline else None
        agent_str = f"{agent_val:{fmt}}"
        base_str  = f"{base_val:{fmt}}" if base_val is not None else "    n/a"
        match = " ✓" if base_val is not None and abs(agent_val - base_val) < 0.001 else ""
        print(f"  {label:<32} {agent_str:>10}  {base_str:>12}{match}")

    row("F1 score",            "f1")
    row("ROC-AUC",             "roc_auc")
    row("Avg precision",       "avg_precision")
    row("Precision",           "precision")
    row("Recall",              "recall")

    print(f"\n  {'Confusion matrix':}")
    print(f"  {'  TP (correctly flagged churners)':<38} {m['true_positives']:>6}")
    print(f"  {'  FP (non-churners we intervene on)':<38} {m['false_positives']:>6}")
    print(f"  {'  FN (churners we miss)':<38} {m['false_negatives']:>6}")
    print(f"  {'  TN (low-risk, correctly ignored)':<38} {m['true_negatives']:>6}")

    print(f"\n  {'Intervention stats':}")
    print(f"  {'  War rooms fired':<38} {m['interventions_fired']:>6}  of 1409 customers")
    print(f"  {'  Actual churners in test set':<38} {m['actual_churners']:>6}")

    print(f"\n  {'Business impact (monthly $)'}")
    print(f"  {'  Revenue at risk — caught':<38} ${m['revenue_at_risk_caught_monthly']:>8,.2f}")
    print(f"  {'  Revenue at risk — missed':<38} ${m['revenue_missed_monthly']:>8,.2f}")
    print(f"  {'  Non-churner charges flagged':<38} ${m['false_positive_spend_monthly']:>8,.2f}")

    catch_rate = m["revenue_at_risk_caught_monthly"] / (
        m["revenue_at_risk_caught_monthly"] + m["revenue_missed_monthly"] + 1e-9
    )
    print(f"  {'  Revenue catch rate':<38}  {catch_rate:>7.1%}")

    print(f"\n  Note: Agent F1 ≈ ML baseline F1 by design — both use")
    print(f"  xgb_pipeline.pkl + threshold={m['threshold']:.2f}. The business")
    print(f"  metrics above are what the pure ML report cannot show.")
    print(sep)


def main():
    print("Loading test set...")
    eng_test, X_test, y_test = load_test_split()
    monthly_charges = eng_test["Monthly Charges"].values

    print(f"Scoring all {len(X_test)} test customers (no LLM, XGBoost only)...")
    y_prob = score_all_customers(X_test)

    threshold_cfg = joblib.load(MODELS_DIR / "threshold.pkl")
    threshold     = threshold_cfg["best_threshold"]

    metrics  = compute_metrics(y_test.values, y_prob, threshold, monthly_charges)
    baseline = load_ml_baseline()

    print_report(metrics, baseline)

    # Save
    out_path = REPORTS_DIR / "agent_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()