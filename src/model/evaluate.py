import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score,
)

REPORTS_DIR = Path("models/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(y_test, y_pred, y_prob) -> dict:
    print("\n-- Classification report ---")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    auc = roc_auc_score(y_test, y_prob)
    ap  = average_precision_score(y_test, y_prob)
    print(f"  ROC-AUC:        {auc:.4f}")
    print(f"  Avg Precision:  {ap:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"XGBoost (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curve"); ax.legend()
    fig.savefig(REPORTS_DIR / "roc_curve.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, lw=2, label=f"XGBoost (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("PR Curve"); ax.legend()
    fig.savefig(REPORTS_DIR / "pr_curve.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    metrics = {
        "roc_auc": round(float(auc), 4),
        "average_precision": round(float(ap), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
    }
    with open(REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Reports saved to {REPORTS_DIR}")
    return metrics