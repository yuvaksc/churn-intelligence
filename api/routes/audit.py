"""api/routes/audit.py — read the append-only audit trail.

GET /api/audit              — most recent analyses (newest first)
GET /api/audit/{trace_id}   — full reconstructable record for one analysis
"""

import asyncio

from fastapi import APIRouter, HTTPException, Query

from db import audit as db_audit

router = APIRouter(tags=["audit"])


@router.get("/audit")
async def list_audit(limit: int = Query(20, ge=1, le=200)):
    return await asyncio.to_thread(db_audit.list_audit, limit)


@router.get("/audit/{trace_id}")
async def get_audit(trace_id: str):
    row = await asyncio.to_thread(db_audit.get_audit, trace_id)
    if row is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return row
