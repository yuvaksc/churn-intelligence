"""api/routes/logs.py — GET /api/logs"""

import os
import json
from pathlib import Path

from fastapi import APIRouter, Query
from api.schemas import RetentionLogEntry

router   = APIRouter(tags=["logs"])
LOG_PATH = Path("models/reports/retention_log.jsonl")


@router.get("/logs", response_model=list[RetentionLogEntry])
async def get_logs(
    limit:  int = Query(20, ge=1, le=200),
    status: str = Query("", description="Filter by status e.g. PENDING_CONTACT"),
):
    """Returns the most recent retention log entries, newest first."""
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return _get_logs_dynamodb(limit, status)
    return _get_logs_file(limit, status)


def _get_logs_dynamodb(limit, status):
    import boto3
    from boto3.dynamodb.conditions import Key
    table_name = os.environ.get("DYNAMODB_TABLE", "churn-prod-main")
    region = os.environ.get("BEDROCK_REGION", "us-east-1")
    ddb = boto3.resource("dynamodb", region_name=region)
    table = ddb.Table(table_name)

    from datetime import datetime, timedelta
    entries = []
    # Query the log-date-index GSI for the last 7 days, newest first
    for days_back in range(0, 7):
        day = (datetime.now() - timedelta(days=days_back)).isoformat()[:10]
        resp = table.query(
            IndexName="log-date-index",
            KeyConditionExpression=Key("log_date").eq(day),
            ScanIndexForward=False,
        )
        for item in resp.get("Items", []):
            if status and item.get("status") != status:
                continue
            item["risk_score"] = float(item.get("risk_score", 0))
            item["monthly_charge"] = float(item.get("monthly_charge", 0))
            try:
                entries.append(RetentionLogEntry(**{
                    k: v for k, v in item.items()
                    if k in RetentionLogEntry.model_fields
                }))
            except Exception:
                continue
            if len(entries) >= limit:
                return entries
    return entries


def _get_logs_file(limit, status):
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    entries = []
    for line in reversed(lines):
        try:
            entry = json.loads(line)
            if status and entry.get("status") != status:
                continue
            entries.append(RetentionLogEntry(**entry))
            if len(entries) >= limit:
                break
        except Exception:
            continue
    return entries