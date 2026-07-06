"""API route check — the logs endpoint over the real SQLite layer, via FastAPI's
TestClient. Mounted in isolation (just this router) so no ML stack is imported."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import logs
from db import recommendations as R

app = FastAPI()
app.include_router(logs.router, prefix="/api")
client = TestClient(app)


def _rec(log_id="RET-1", customer_id="TEST-1"):
    return {
        "log_id": log_id, "customer_id": customer_id, "risk_score": 0.9,
        "offer_text": "10% loyalty discount", "contract_type": "Month-to-month",
        "monthly_charge": 50.0, "timestamp": "2026-01-01T00:00:00",
        "status": R.DEFAULT_STATUS, "assigned_to": R.DEFAULT_ASSIGNED_TO,
    }


def test_logs_empty(db):
    r = client.get("/api/logs")
    assert r.status_code == 200
    assert r.json() == []


def test_logs_returns_inserted_recommendation(db):
    R.insert_recommendation(_rec())
    r = client.get("/api/logs")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["log_id"] == "RET-1"


def test_logs_limit_is_validated(db):
    assert client.get("/api/logs?limit=0").status_code == 422     # ge=1
    assert client.get("/api/logs?limit=999").status_code == 422   # le=200
