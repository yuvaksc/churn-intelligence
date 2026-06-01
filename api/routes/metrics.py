"""api/routes/metrics.py — GET /api/metrics"""

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from api.schemas import MetricsResponse
from api.dependencies import AppState, get_state

router      = APIRouter(tags=["metrics"])
METRICS_PATH = Path("models/reports/agent_evaluation.json")


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(state: AppState = Depends(get_state)):
    """
    Returns model evaluation metrics.
    Reads from agent_evaluation.json if it exists (run evaluate_agents.py first).
    Falls back to computing precision/recall from pre-scored test set.
    """
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            data = json.load(f)
        return MetricsResponse(**data)

    # Fallback: compute from pre-scored state
    import numpy as np
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
        roc_auc_score, average_precision_score,
        confusion_matrix,
    )

    y_true = state.y_test.values
    y_prob = state.risk_scores
    y_pred = (y_prob >= state.threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    eng = state.eng_test_raw
    monthly = eng["Monthly Charges"].values

    RETENTION_SUCCESS = 0.30
    AVG_LTR           = 24
    DISCOUNT_CREDIT   = 20.0
    DISCOUNT_PCT      = 0.10
    DISCOUNT_MONTHS   = 6

    tp_mask = (y_pred == 1) & (y_true == 1)
    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)

    tp_monthly = monthly[tp_mask]
    fp_monthly = monthly[fp_mask]
    fn_monthly = monthly[fn_mask]

    tp_value  = int(tp) * RETENTION_SUCCESS * (float(tp_monthly.mean()) if len(tp_monthly) else 0) * AVG_LTR
    fp_cost   = int(fp) * (DISCOUNT_CREDIT + (float(fp_monthly.mean()) if len(fp_monthly) else 0) * DISCOUNT_PCT * DISCOUNT_MONTHS)
    fn_missed = float(fn_monthly.sum())

    return MetricsResponse(
        threshold=round(float(state.threshold), 4),
        f1=round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        precision=round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        recall=round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        roc_auc=round(float(roc_auc_score(y_true, y_prob)), 4),
        avg_precision=round(float(average_precision_score(y_true, y_prob)), 4),
        true_positives=int(tp),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_negatives=int(tn),
        interventions_fired=int(y_pred.sum()),
        actual_churners=int(y_true.sum()),
        revenue_at_risk_caught_monthly=round(tp_value, 2),
        revenue_missed_monthly=round(fn_missed, 2),
        false_positive_spend_monthly=round(fp_cost, 2),
    )