"""
run_pipeline.py -- master entrypoint.

Usage:
    python run_pipeline.py

Model: XGBoost (single model, scale_pos_weight, GridSearchCV)
Artefacts: models/xgboost_churn.pkl + models/threshold.pkl
"""

import sys
import traceback
from src.data.loader import load_and_prepare
from src.model.train import train_all
from src.model.evaluate import evaluate
from src.explainability.shap_explainer import run_explainability


def main():
    print("=" * 52)
    print("  Churn Intelligence -- ML Pipeline")
    print("  Model: XGBoost (IBM Telco, honest leakage removal)")
    print("=" * 52)

    print("\n[1/4] Loading and preparing data...")
    data = load_and_prepare()

    print("\n[2/4] Training XGBoost...")
    results = train_all(data)

    print("\n[3/4] Evaluating...")
    metrics = evaluate(
        data["y_test"],
        results["y_pred_test"],
        results["y_prob_test"],
    )

    print("\n[4/4] Running SHAP explainability...")
    run_explainability(
        results["model"],
        data["gbm"]["X_train"],
        data["gbm"]["X_test"],
    )

    print("\n" + "=" * 52)
    print("  FINAL RESULTS")
    print("=" * 52)
    print(f"  F1:             {results['final_f1']:.4f}")
    print(f"  ROC-AUC:        {metrics['roc_auc']}")
    print(f"  Avg Precision:  {metrics['average_precision']}")
    print(f"  Threshold:      {results['best_threshold']}  (tuned on val set)")
    print()
    print("  Artefacts saved:")
    print("    models/xgboost_churn.pkl     <- inference model")
    print("    models/threshold.pkl         <- threshold config")
    print("    models/shap_explainer.pkl    <- SHAP explainer")
    print("    models/shap_values_test.csv  <- per-customer SHAP")
    print("    models/reports/              <- charts + metrics.json")
    print()
    print("  Next: python run_agents.py")
    print("=" * 52)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)