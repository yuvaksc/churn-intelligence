"""WebSocket route check — the war-room stream's connection + not-found handling.

api/routes/ws.py imports the full LangGraph war room and the numpy/pandas-backed
customers module at import time. Those are stubbed here so the route imports without
the ML stack; the full graph run is integration-tested via Docker. We only assert the
WebSocket contract: connect → unknown customer → an `error` frame."""

import sys
import types
from unittest.mock import MagicMock

# ── Stub ws.py's heavy module-level imports BEFORE importing the route ────────────
sys.modules["agents.graph"] = types.SimpleNamespace(war_room_graph=MagicMock())

import db                                              # noqa: E402  (light package)
_fake_customers = types.ModuleType("db.customers")
_fake_customers.get_customer_features = lambda cid: None   # any id → "not in test set"
_fake_customers.get_customer = lambda cid: None
db.customers = _fake_customers
sys.modules["db.customers"] = _fake_customers

sys.modules["eval"] = types.SimpleNamespace(
    experiment_callbacks=lambda *a, **k: [],
    score_run=lambda *a, **k: None,
)

from fastapi import FastAPI                            # noqa: E402
from fastapi.testclient import TestClient              # noqa: E402
from api.routes import ws                              # noqa: E402

app = FastAPI()
app.include_router(ws.router, prefix="/api")
client = TestClient(app)


def test_ws_unknown_customer_returns_error_frame():
    with client.websocket_connect("/api/analyze/999/ws") as sock:
        msg = sock.receive_json()
    assert msg["type"] == "error"
    assert "999" in msg["message"]
