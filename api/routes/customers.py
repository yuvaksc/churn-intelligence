"""
api/routes/customers.py

GET /api/customers              — paginated list, sorted by risk_score desc
GET /api/customers/{customer_id} — full feature set + pre-computed SHAP
"""

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from api.schemas import CustomerSummary, CustomerDetail, ShapDriver
from api.dependencies import AppState, get_state, get_customer_raw, get_top_shap_drivers

router = APIRouter(tags=["customers"])


def _build_summary(idx: int, state: AppState) -> CustomerSummary:
    pos       = state.eng_test_raw.index.get_loc(idx)
    score     = float(state.risk_scores[pos])
    row       = state.eng_test_raw.loc[idx]
    true_lbl  = int(state.y_test.loc[idx]) if idx in state.y_test.index else None

    return CustomerSummary(
        customer_id=idx,
        risk_score=round(score, 4),
        risk_label="HIGH" if score >= state.threshold else "LOW",
        contract=str(row.get("Contract", "")),
        monthly_charges=float(row.get("Monthly Charges", 0)),
        tenure_months=int(row.get("Tenure Months", 0)),
        internet_service=str(row.get("Internet Service", "")),
        services_count=int(row.get("Services Count", 0)),
        high_risk_flag=int(row.get("High Risk Flag", 0)),
        true_label=true_lbl,
    )


@router.get("/customers", response_model=list[CustomerSummary])
async def list_customers(
    limit:     int  = Query(50,    ge=1, le=500),
    offset:    int  = Query(0,     ge=0),
    risk_only: bool = Query(False, description="Only return HIGH risk customers"),
    state: AppState = Depends(get_state),
):
    """
    Returns test customers sorted by risk score descending.
    Use risk_only=true to filter to only HIGH risk customers.
    """
    indices = state.eng_test_raw.index.tolist()
    scores  = state.risk_scores

    # Sort by risk score descending
    sorted_pairs = sorted(
        zip(indices, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    if risk_only:
        sorted_pairs = [(i, s) for i, s in sorted_pairs if s >= state.threshold]

    page = sorted_pairs[offset : offset + limit]
    return [_build_summary(idx, state) for idx, _ in page]


@router.get("/customers/{customer_id}", response_model=CustomerDetail)
async def get_customer(
    customer_id: int,
    state: AppState = Depends(get_state),
):
    """Returns full feature set and top SHAP drivers for one customer."""
    customer_raw = get_customer_raw(customer_id, state)
    if customer_raw is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not in test set")

    summary      = _build_summary(customer_id, state)
    shap_drivers = get_top_shap_drivers(customer_id, state)

    return CustomerDetail(
        **summary.model_dump(),
        all_features=customer_raw,
        top_shap_drivers=[ShapDriver(**d) for d in shap_drivers],
    )