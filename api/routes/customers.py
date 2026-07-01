"""
api/routes/customers.py

GET /api/customers               — paginated list, sorted by risk_score desc
GET /api/customers/{customer_id} — full feature set + pre-computed SHAP

Both read straight from the SQLite `customers` table (seeded at startup).
"""

import asyncio

from fastapi import APIRouter, HTTPException, Query

from api.schemas import CustomerSummary, CustomerDetail
from db import customers as db_customers

router = APIRouter(tags=["customers"])


@router.get("/customers", response_model=list[CustomerSummary])
async def list_customers(
    limit:     int  = Query(50,    ge=1, le=500),
    offset:    int  = Query(0,     ge=0),
    risk_only: bool = Query(False, description="Only return HIGH risk customers"),
):
    """
    Returns test customers sorted by risk score descending.
    Use risk_only=true to filter to only HIGH risk customers.
    """
    return await asyncio.to_thread(db_customers.list_customers, limit, offset, risk_only)


@router.get("/customers/{customer_id}", response_model=CustomerDetail)
async def get_customer(customer_id: int):
    """Returns full feature set and top SHAP drivers for one customer."""
    detail = await asyncio.to_thread(db_customers.get_customer, customer_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not in test set")
    return detail
