"""WebSocket route: rate limiting, unknown customers, and a full streamed run with the
war-room graph replaced by a scripted event stream (no LLM, no database)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import ratelimit
from api.routes import ws


class _Customers:
    def features(self, customer_id):
        return {"Contract": "Month-to-month", "Monthly Charges": 80.0} if customer_id == 1 else None

    def get(self, customer_id):
        return {"risk_label": "HIGH", "top_shap_drivers": []} if customer_id == 1 else None


class _Recs:
    rows: list = []

    def insert(self, rec):
        self.rows.append(rec)


class _FakeGraph:
    """Emits the same event shapes LangGraph's astream_events(version="v2") does."""

    async def astream_events(self, initial, version, config):
        yield {"event": "on_chain_end", "name": "agent1", "data": {"output": {
            "risk_score": 0.9, "risk_label": "HIGH", "shap_drivers": [], "risk_summary": "High risk."}}}
        yield {"event": "on_chain_end", "name": "agent3", "data": {"output": {
            "retention_offer": "HOOK: hi\nOFFER: 10% off for 6 months.\nURGENCY: this week.",
            "policy": {"max_discount_pct": 25}, "crm_log_id": "RET-1", "crm_logged": True, "tools_used": []}}}


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(ws, "get_customer_repo", lambda: _Customers())
    monkeypatch.setattr(ws, "get_recommendation_repo", lambda: _Recs())
    monkeypatch.setattr(ws, "war_room_graph", _FakeGraph())
    monkeypatch.setattr(ws, "tracing_on", lambda: False)
    monkeypatch.setattr(ws, "analysis_limiter", ratelimit.SlidingWindowLimiter(100))
    _Recs.rows = []
    app = FastAPI()
    app.include_router(ws.router, prefix="/api")
    return TestClient(app)


def _frames(client, url):
    out = []
    with client.websocket_connect(url) as sock:
        while True:
            m = sock.receive_json()
            out.append(m)
            if m["type"] in ("done", "error"):
                return out


def test_unknown_customer_returns_error_frame(client):
    frames = _frames(client, "/api/analyze/999/ws")
    assert frames[-1]["type"] == "error" and "999" in frames[-1]["message"]


def test_full_run_streams_agents_applies_guardrails_and_persists(client):
    frames = _frames(client, "/api/analyze/1/ws")
    completes = [f["agent"] for f in frames if f["type"] == "agent_complete"]
    assert completes == ["agent1", "agent3"]
    agent3 = next(f for f in frames if f.get("agent") == "agent3" and f["type"] == "agent_complete")
    assert agent3["data"]["guardrails"]["passed"] is True
    assert frames[-1] == {"type": "done", "risk_label": "HIGH", "input_scan": frames[-1]["input_scan"]}
    assert _Recs.rows[0]["log_id"] == "RET-1" and _Recs.rows[0]["status"] == "PENDING_CONTACT"


def test_rate_limit_closes_with_error(client, monkeypatch):
    monkeypatch.setattr(ws, "analysis_limiter", ratelimit.SlidingWindowLimiter(1))
    _frames(client, "/api/analyze/1/ws")
    frames = _frames(client, "/api/analyze/1/ws")
    assert frames[-1]["type"] == "error" and "Rate limit" in frames[-1]["message"]
