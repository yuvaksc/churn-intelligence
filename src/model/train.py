"""
Simple XGBoost — single model, scale_pos_weight for imbalance, GridSearchCV.

Why this over everything we tried:
- Same F1 (~0.63) and ROC-AUC (~0.84) as the ensemble
- Single model = single .pkl file on AWS Lambda
- No SMOTE overhead = fast inference
- GridSearchCV over a tight grid = reproducible, no Optuna dependency
- Matches the TDS article's approach on our exact dataset

AWS Lambda friendly:
- xgboost is ~60MB installed, fits in a Lambda layer
- predict_proba on one row: <5ms
"""

import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def run_baseline(X_train, X_test, y_train, y_test) -> dict:
    print("\n-- Baseline: Logistic Regression ---")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )),
    ])
    pipe.fit(X_train, y_train)
    f1 = f1_score(y_test, pipe.predict(X_test))
    print(f"  Logistic Regression   F1: {f1:.4f}")
    return {"Logistic Regression": {"f1": f1}}


def train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test) -> dict:
    import pandas as pd

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = round(neg / pos, 2)
    print(f"\n-- XGBoost (scale_pos_weight={spw}) ---")

    # Tight grid — covers the best param regions from all our Optuna runs
    param_grid = {
        "n_estimators":     [200, 400],
        "max_depth":        [3, 4],
        "learning_rate":    [0.05, 0.1],
        "subsample":        [0.8, 0.9],
        "colsample_bytree": [0.8, 0.9],
        "min_child_weight": [3, 5],
        "reg_lambda":       [1, 2],
        "gamma":            [0, 0.1],
    }

    base = XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    search = GridSearchCV(
        base, param_grid,
        cv=CV, scoring="f1",
        n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"Best CV F1:     {search.best_score_:.4f}")
    print(f"Best params:    {search.best_params_}")

    # Refit best params on train+val (80%)
    X_tv = pd.concat([X_train, X_val])
    y_tv = pd.concat([y_train, y_val])
    print(f"\nRefitting on train+val: {len(X_tv)} rows...")

    final_model = XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        **search.best_params_,
    )
    final_model.fit(X_tv, y_tv)

    # Tune threshold on val set
    y_val_prob = final_model.predict_proba(X_val)[:, 1]
    best_t, best_val_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(y_val, (y_val_prob >= t).astype(int))
        if f1 > best_val_f1:
            best_val_f1, best_t = f1, t
    print(f"Val threshold:  {best_t:.2f}  (val F1={best_val_f1:.4f})")

    # Evaluate once on test
    y_test_prob = final_model.predict_proba(X_test)[:, 1]
    y_pred      = (y_test_prob >= best_t).astype(int)
    test_f1     = f1_score(y_test, y_pred)

    print(f"\n-- Classification report (test) ---")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return {
        "model":          final_model,
        "best_threshold": best_t,
        "f1":             test_f1,
        "y_pred":         y_pred,
        "y_prob":         y_test_prob,
    }


def train_all(data: dict) -> dict:
    gbm     = data["gbm"]
    y_train = data["y_train"]
    y_val   = data["y_val"]
    y_test  = data["y_test"]

    baseline = run_baseline(
        gbm["X_train"], gbm["X_test"], y_train, y_test
    )

    result = train_xgboost(
        gbm["X_train"], gbm["X_val"], gbm["X_test"],
        y_train, y_val, y_test,
    )

    print("\n" + "=" * 52)
    print("  SUMMARY")
    print("  " + "-" * 38)
    print(f"  {'Logistic Regression':<30} {baseline['Logistic Regression']['f1']:>6.4f}")
    print(f"  {'XGBoost (tuned)':<30} {result['f1']:>6.4f}  << FINAL")
    print("=" * 52)

    # Save — only two files needed at inference time
    joblib.dump(result["model"], MODELS_DIR / "xgboost_churn.pkl")
    joblib.dump({
        "best_threshold": result["best_threshold"],
        "scale_pos_weight": round((y_train == 0).sum() / (y_train == 1).sum(), 2),
    }, MODELS_DIR / "threshold.pkl")

    print("\nSaved:")
    print("  models/xgboost_churn.pkl    ← inference model")
    print("  models/threshold.pkl        ← threshold config")
    print("  models/shap_explainer.pkl   ← built in step 4")

    return {
        "model":           result["model"],
        "lgbm":            None,
        "xgb":             result["model"],
        "best_threshold":  result["best_threshold"],
        "final_f1":        result["f1"],
        "y_pred_test":     result["y_pred"],
        "y_prob_test":     result["y_prob"],
        "baselines":       baseline,
    }