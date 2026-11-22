"""churn_ml.training — baseline vs tuned XGBoost, calibration, economic threshold.

Splits (stratified, seed 42, identical rows to the original pipeline):
    train 60%  → model fitting + 5-fold stratified CV tuning
    val   20%  → split in two: calibration half (isotonic) / threshold half
    test  20%  → touched once, for the reported numbers and champion comparison

The threshold is chosen on calibrated probabilities from the economics in
churn_ml.economics, on data the calibrator never saw.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from churn_data.loader import clean, load_raw, three_way_split
from churn_ml import economics
from churn_ml.model import RAW_FEATURES, TARGET, logistic_baseline, xgboost_pipeline

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

XGB_SEARCH_SPACE = {
    "clf__n_estimators": randint(150, 500),
    "clf__max_depth": randint(2, 6),
    "clf__learning_rate": loguniform(0.02, 0.2),
    "clf__subsample": uniform(0.7, 0.3),
    "clf__colsample_bytree": uniform(0.6, 0.4),
    "clf__min_child_weight": randint(1, 8),
    "clf__reg_lambda": loguniform(0.5, 5),
    "clf__gamma": uniform(0, 0.3),
}


@dataclass
class Splits:
    X_train: pd.DataFrame
    X_cal: pd.DataFrame
    X_thr: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_cal: pd.Series
    y_thr: pd.Series
    y_test: pd.Series


@dataclass
class TrainResult:
    model: CalibratedClassifierCV
    threshold: float
    metrics: dict
    baseline_metrics: dict
    params: dict
    version: str
    data_fingerprint: str
    splits: Splits = field(repr=False)


def load_training_frame() -> pd.DataFrame:
    df = clean(load_raw())
    return df[RAW_FEATURES + [TARGET]]


def make_splits(df: pd.DataFrame) -> Splits:
    X_tr, X_val, X_te, y_tr, y_val, y_te = three_way_split(df, target=TARGET)
    X_cal, X_thr, y_cal, y_thr = train_test_split(X_val, y_val, test_size=0.5,
                                                  random_state=42, stratify=y_val)
    return Splits(X_tr, X_cal, X_thr, X_te, y_tr, y_cal, y_thr, y_te)


def evaluate(p: np.ndarray, y: pd.Series, monthly: pd.Series, threshold: float,
             p_raw: np.ndarray | None = None) -> dict:
    pred = (p >= threshold).astype(int)
    out = {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "flagged_rate": float(pred.mean()),
        "net_value_usd": economics.net_value(y, pred.astype(bool), monthly),
        "threshold": threshold,
    }
    if p_raw is not None:
        out["brier_uncalibrated"] = brier_score_loss(y, p_raw)
        out["roc_auc_uncalibrated"] = roc_auc_score(y, p_raw)
    return {k: round(float(v), 4) for k, v in out.items()}


def _calibrate(fitted, s: Splits) -> CalibratedClassifierCV:
    # cv="prefit": the model is already fitted on train; isotonic maps its scores
    # to probabilities using the calibration half of validation only.
    return CalibratedClassifierCV(fitted, method="isotonic", cv="prefit").fit(s.X_cal, s.y_cal)


def train(df: pd.DataFrame | None = None, n_iter: int = 25, seed: int = 42) -> TrainResult:
    df = load_training_frame() if df is None else df
    s = make_splits(df)
    spw = round(float((s.y_train == 0).sum() / (s.y_train == 1).sum()), 3)

    # ── Baseline: logistic regression, same preprocessing, same calibration ──
    base = _calibrate(logistic_baseline().fit(s.X_train, s.y_train), s)
    base_thr, _ = economics.choose_threshold(s.y_thr, base.predict_proba(s.X_thr)[:, 1],
                                             s.X_thr["Monthly Charges"])
    base_raw = base.estimator.predict_proba(s.X_test)[:, 1]
    baseline_metrics = evaluate(base.predict_proba(s.X_test)[:, 1], s.y_test,
                                s.X_test["Monthly Charges"], base_thr, base_raw)

    # ── XGBoost: stratified 5-fold CV random search on train ─────────────────
    search = RandomizedSearchCV(xgboost_pipeline(spw), XGB_SEARCH_SPACE, n_iter=n_iter,
                                scoring="roc_auc", cv=CV, random_state=seed, n_jobs=-1, refit=True)
    search.fit(s.X_train, s.y_train)
    model = _calibrate(search.best_estimator_, s)

    p_thr = model.predict_proba(s.X_thr)[:, 1]
    threshold, _ = economics.choose_threshold(s.y_thr, p_thr, s.X_thr["Monthly Charges"])
    p_test_raw = search.best_estimator_.predict_proba(s.X_test)[:, 1]
    metrics = evaluate(model.predict_proba(s.X_test)[:, 1], s.y_test,
                       s.X_test["Monthly Charges"], threshold, p_test_raw)
    metrics["cv_roc_auc"] = round(float(search.best_score_), 4)

    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:12]
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + fingerprint
    params = {k.removeprefix("clf__"): (round(float(v), 5) if isinstance(v, float) else v)
              for k, v in search.best_params_.items()}
    params["scale_pos_weight"] = spw

    return TrainResult(model=model, threshold=threshold, metrics=metrics,
                       baseline_metrics=baseline_metrics, params=params, version=version,
                       data_fingerprint=fingerprint,
                       splits=s)


def model_card(r: TrainResult) -> str:
    m, b = r.metrics, r.baseline_metrics
    rows = "\n".join(
        f"| {k} | {b.get(k, '')} | {m.get(k, '')} |"
        for k in ["roc_auc", "pr_auc", "brier", "recall", "precision", "f1", "net_value_usd", "threshold"]
    )
    return f"""# Model card — churn-xgb {r.version}

**Task:** predict 30-day churn for telecom customers; output is a calibrated probability.
**Data:** IBM Telco Customer Churn (7,043 customers), fingerprint `{r.data_fingerprint}`.
Split 60/20/20 stratified (seed 42); validation halved into calibration / threshold sets.

## Model
sklearn Pipeline: FeatureEngineer (medians learned at fit) → ColumnTransformer →
XGBoost (5-fold stratified CV random search on ROC-AUC) → isotonic calibration.

Best params: `{json.dumps(r.params)}`

## Test-set results (vs. logistic-regression baseline, same pipeline + calibration)
| metric | baseline (LR) | XGBoost |
|---|---|---|
{rows}

Calibration: Brier {m.get('brier_uncalibrated')} uncalibrated → {m['brier']} calibrated.

## Decision threshold
{r.threshold:.2f}, chosen to maximise realised retention net value on the threshold
half of validation, using calibrated probabilities and these assumptions:
`{json.dumps(economics.ASSUMPTIONS)}`.

## Intended use and limits
Ranks customers for retention outreach by analysts; not for automated pricing.
Trained on one public dataset from one market: re-validate on each client's data
(champion/challenger comparison) before relying on the threshold.
SHAP explanations describe the uncalibrated XGBoost margin; isotonic calibration is
monotonic, so driver rankings are unaffected.
"""
