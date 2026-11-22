"""
api/routes/customers.py

GET /api/customers               — paginated list, sorted by risk_score desc
GET /api/customers/{customer_id} — full feature set + precomputed SHAP drivers
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_customer_repo
from api.schemas import CustomerDetail, CustomerSummary
from churn_data.ports import CustomerRepository

router = APIRouter(tags=["customers"])


@router.get("/customers", response_model=list[CustomerSummary])
async def list_customers(
    limit:     int  = Query(50,    ge=1, le=500),
    offset:    int  = Query(0,     ge=0),
    risk_only: bool = Query(False, description="Only return HIGH risk customers"),
    repo: CustomerRepository = Depends(get_customer_repo),
):
    """Customers sorted by risk score, highest first."""
    return await asyncio.to_thread(repo.list, limit, offset, risk_only)


@router.get("/customers/{customer_id}", response_model=CustomerDetail)
async def get_customer(customer_id: int, repo: CustomerRepository = Depends(get_customer_repo)):
    """Full feature set and top SHAP drivers for one customer."""
    detail = await asyncio.to_thread(repo.get, customer_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not in test set")
    return detail
