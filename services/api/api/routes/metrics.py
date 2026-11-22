# """api/routes/metrics.py — GET /api/metrics"""

# import json
# import asyncio
# from pathlib import Path

# from fastapi import APIRouter

# from api.schemas import MetricsResponse
# from churn_data.db import customers as db_customers

# router       = APIRouter(tags=["metrics"])
# METRICS_PATH = Path("models/reports/agent_evaluation.json")


# @router.get("/metrics", response_model=MetricsResponse)
# async def get_metrics():
#     """
#     Returns model evaluation metrics.
#     Reads from agent_evaluation.json if it exists (run `python -m churn_ml.holdout_report` first).
#     Falls back to computing metrics from the pre-scored customers table.
#     """
#     if METRICS_PATH.exists():
#         with open(METRICS_PATH) as f:
#             data = json.load(f)
#         return MetricsResponse(**data)

#     return await asyncio.to_thread(_compute_metrics)


# def _compute_metrics() -> MetricsResponse:
#     """Fallback: compute precision/recall/ROI from the pre-scored customers table."""
#     import numpy as np
#     from sklearn.metrics import (
#         f1_score, precision_score, recall_score,
#         roc_auc_score, average_precision_score, confusion_matrix,
#     )

#     rows      = db_customers.fetch_scoring_rows()
#     threshold = db_customers.get_threshold()

#     y_true  = np.array([r["true_label"] for r in rows])
#     y_prob  = np.array([r["risk_score"] for r in rows])
#     monthly = np.array([r["monthly_charges"] for r in rows])
#     y_pred  = (y_prob >= threshold).astype(int)

#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

#     RETENTION_SUCCESS = 0.30
#     AVG_LTR           = 24
#     DISCOUNT_CREDIT   = 20.0
#     DISCOUNT_PCT      = 0.10
#     DISCOUNT_MONTHS   = 6

#     tp_mask = (y_pred == 1) & (y_true == 1)
#     fp_mask = (y_pred == 1) & (y_true == 0)
#     fn_mask = (y_pred == 0) & (y_true == 1)

#     tp_monthly = monthly[tp_mask]
#     fp_monthly = monthly[fp_mask]
#     fn_monthly = monthly[fn_mask]

#     tp_value  = int(tp) * RETENTION_SUCCESS * (float(tp_monthly.mean()) if len(tp_monthly) else 0) * AVG_LTR
#     fp_cost   = int(fp) * (DISCOUNT_CREDIT + (float(fp_monthly.mean()) if len(fp_monthly) else 0) * DISCOUNT_PCT * DISCOUNT_MONTHS)
#     fn_missed = float(fn_monthly.sum())

#     avg = lambda a: float(a.mean()) if len(a) else 0.0

#     tp_value      = int(tp) * RETENTION_SUCCESS * avg(tp_monthly) * AVG_LTR
#     tp_offer_cost = int(tp) * (DISCOUNT_CREDIT + avg(tp_monthly) * DISCOUNT_PCT * DISCOUNT_MONTHS)
#     fp_cost       = int(fp) * (DISCOUNT_CREDIT + avg(fp_monthly) * DISCOUNT_PCT * DISCOUNT_MONTHS)
#     fn_missed     = int(fn) * RETENTION_SUCCESS * avg(fn_monthly) * AVG_LTR   # same basis as tp_value
#     net_value     = tp_value - tp_offer_cost - fp_cost

#     return MetricsResponse(
#         threshold=round(float(threshold), 4),
#         f1=round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
#         precision=round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
#         recall=round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
#         roc_auc=round(float(roc_auc_score(y_true, y_prob)), 4),
#         avg_precision=round(float(average_precision_score(y_true, y_prob)), 4),
#         true_positives=int(tp),
#         false_positives=int(fp),
#         false_negatives=int(fn),
#         true_negatives=int(tn),
#         interventions_fired=int(y_pred.sum()),
#         actual_churners=int(y_true.sum()),
#         revenue_at_risk_caught_monthly=round(tp_value, 2),
#         revenue_missed_monthly=round(fn_missed, 2),
#         false_positive_spend_monthly=round(fp_cost, 2),
#     )











"""api/routes/metrics.py — GET /api/metrics

All dollar figures are modeled, one-time LTV-basis values (NOT monthly revenue),
computed under the assumptions listed in ASSUMPTIONS and returned with the response.
"""

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Depends

from api.deps import get_customer_repo
from api.schemas import MetricsResponse
from churn_data.ports import CustomerRepository

router       = APIRouter(tags=["metrics"])
METRICS_PATH = Path("models/reports/agent_evaluation.json")

# ── Business assumptions: single source of truth shared with threshold selection ──
from churn_ml.economics import (  # noqa: E402
    ASSUMPTIONS, DISCOUNT_CREDIT, DISCOUNT_MONTHS, DISCOUNT_PCT, RETENTION_SUCCESS,
    LTV_MONTHS as AVG_LTR,
)


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(repo: CustomerRepository = Depends(get_customer_repo)):
    """
    Returns model evaluation metrics, computed live from the pre-scored
    customers table so the dollar figures always use the current formulas.

    agent_evaluation.json is only a fallback (e.g. if the DB is unavailable),
    and must have been regenerated with the new field names to validate.
    """
    try:
        return await asyncio.to_thread(_compute_metrics, repo)
    except Exception:
        if METRICS_PATH.exists():
            with open(METRICS_PATH) as f:
                return MetricsResponse(**json.load(f))
        raise


def _mean(a) -> float:
    return float(a.mean()) if len(a) else 0.0


# Plain-numpy versions of the scikit-learn metrics (identical results; the serverless API
# does not ship scikit-learn).

def _ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _roc_auc(y_true, y_prob) -> float:
    """Probability a random churner outranks a random non-churner (ties count half)."""
    import numpy as np
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if not n_pos or not n_neg:
        return 0.0
    _, inverse, counts = np.unique(y_prob, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2          # 1-based, ties averaged
    ranks = avg_rank[inverse]
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_precision(y_true, y_prob) -> float:
    """Area under the precision-recall curve, step-wise (scikit-learn's definition)."""
    import numpy as np
    if not y_true.sum():
        return 0.0
    order = np.argsort(-y_prob, kind="mergesort")
    y, s = y_true[order], y_prob[order]
    cut = np.r_[np.where(np.diff(s))[0], len(y) - 1]         # last index of each score
    tps = np.cumsum(y)[cut]
    precision = tps / (cut + 1)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _compute_metrics(repo: CustomerRepository) -> MetricsResponse:
    """Compute classification metrics and LTV-basis business impact."""
    import numpy as np

    # Only labelled rows can be scored (uploaded customers have no ground truth yet).
    rows      = [r for r in repo.scoring_rows() if r.get("true_label") is not None]
    threshold = repo.threshold()

    y_true  = np.array([r["true_label"] for r in rows])
    y_prob  = np.array([r["risk_score"] for r in rows])
    monthly = np.array([r["monthly_charges"] for r in rows])
    y_pred  = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    tp_avg = _mean(monthly[(y_pred == 1) & (y_true == 1)])
    fp_avg = _mean(monthly[(y_pred == 1) & (y_true == 0)])
    fn_avg = _mean(monthly[(y_pred == 0) & (y_true == 1)])

    def offer_cost(avg_monthly: float) -> float:
        return DISCOUNT_CREDIT + avg_monthly * DISCOUNT_PCT * DISCOUNT_MONTHS

    # Value: expected LTV saved among caught churners.
    ltv_saved = tp * RETENTION_SUCCESS * tp_avg * AVG_LTR

    # Value left on the table: SAME basis as ltv_saved (apples to apples).
    ltv_missed = fn * RETENTION_SUCCESS * fn_avg * AVG_LTR

    # Cost: every customer who receives an offer, true positive or not.
    campaign_cost = tp * offer_cost(tp_avg) + fp * offer_cost(fp_avg)
    fp_waste      = fp * offer_cost(fp_avg)   # the wasted share of campaign_cost

    net_value    = ltv_saved - campaign_cost
    return_ratio = (ltv_saved / campaign_cost) if campaign_cost > 0 else 0.0

    return MetricsResponse(
        threshold=round(float(threshold), 4),
        f1=round(_ratio(2 * tp, 2 * tp + fp + fn), 4),
        precision=round(_ratio(tp, tp + fp), 4),
        recall=round(_ratio(tp, tp + fn), 4),
        roc_auc=round(_roc_auc(y_true, y_prob), 4),
        avg_precision=round(_average_precision(y_true, y_prob), 4),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        interventions_fired=int(y_pred.sum()),
        actual_churners=int(y_true.sum()),
        expected_ltv_saved=round(ltv_saved, 2),
        expected_ltv_missed=round(ltv_missed, 2),
        campaign_cost=round(campaign_cost, 2),
        false_positive_waste=round(fp_waste, 2),
        net_value=round(net_value, 2),
        return_ratio=round(return_ratio, 2),
        assumptions=ASSUMPTIONS,
    )