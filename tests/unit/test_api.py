"""REST routes through FastAPI with in-memory repositories (no database).

Routes depend on CustomerRepository / RecommendationRepository via FastAPI Depends,
so the fakes below replace Postgres with `app.dependency_overrides`.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.deps import get_customer_repo, get_recommendation_repo
from api.routes import customers, logs, metrics

CUSTOMERS = [
    {"customer_id": i, "risk_score": s, "risk_label": "HIGH" if s >= 0.5 else "LOW",
     "contract": "Month-to-month", "monthly_charges": 70.0 + i, "tenure_months": i,
     "internet_service": "Fiber optic", "services_count": 2, "high_risk_flag": int(s >= 0.5),
     "true_label": y}
    for i, (s, y) in enumerate([(0.9, 1), (0.8, 1), (0.7, 0), (0.3, 1), (0.1, 0), (0.05, 0)], start=1)
]


class FakeCustomers:
    def list(self, limit, offset, risk_only):
        rows = [c for c in CUSTOMERS if not risk_only or c["risk_label"] == "HIGH"]
        return sorted(rows, key=lambda c: -c["risk_score"])[offset:offset + limit]

    def get(self, customer_id):
        row = next((c for c in CUSTOMERS if c["customer_id"] == customer_id), None)
        return None if row is None else {**row, "all_features": {"Contract": row["contract"]},
                                         "top_shap_drivers": []}

    def features(self, customer_id):
        return (self.get(customer_id) or {}).get("all_features")

    def count(self):
        return len(CUSTOMERS)

    def count_high_risk(self):
        return sum(c["risk_label"] == "HIGH" for c in CUSTOMERS)

    def threshold(self, default=0.5):
        return 0.5

    def scoring_rows(self):
        return [{"risk_score": c["risk_score"], "true_label": c["true_label"],
                 "monthly_charges": c["monthly_charges"]} for c in CUSTOMERS]


class FakeRecs:
    def __init__(self):
        self.rows = []

    def insert(self, rec):
        self.rows.append(rec)

    def list(self, limit=200, status=""):
        return [r for r in reversed(self.rows) if not status or r["status"] == status][:limit]

    def for_customer(self, customer_id, limit=5):
        return [r for r in self.rows if r["customer_id"] == customer_id][:limit]


def _rec(log_id="RET-1"):
    return {"log_id": log_id, "customer_id": "TEST-1", "risk_score": 0.9, "offer_text": "10% loyalty discount",
            "contract_type": "Month-to-month", "monthly_charge": 50.0, "timestamp": "2026-01-01T00:00:00",
            "status": "PENDING_CONTACT", "assigned_to": "Retention Team Queue"}


@pytest.fixture()
def client():
    app = FastAPI()
    for r in (customers, logs, metrics):
        app.include_router(r.router, prefix="/api")
    recs = FakeRecs()
    app.dependency_overrides[get_customer_repo] = lambda: FakeCustomers()
    app.dependency_overrides[get_recommendation_repo] = lambda: recs
    c = TestClient(app)
    c.recs = recs
    return c


def test_customers_sorted_by_risk_and_filterable(client):
    body = client.get("/api/customers?limit=10").json()
    assert [c["customer_id"] for c in body] == [1, 2, 3, 4, 5, 6]
    high = client.get("/api/customers?risk_only=true").json()
    assert {c["risk_label"] for c in high} == {"HIGH"} and len(high) == 3


def test_customer_detail_and_404(client):
    assert client.get("/api/customers/1").json()["all_features"] == {"Contract": "Month-to-month"}
    assert client.get("/api/customers/999").status_code == 404


def test_logs_return_newest_first(client):
    client.recs.insert(_rec("RET-1"))
    client.recs.insert(_rec("RET-2"))
    assert [r["log_id"] for r in client.get("/api/logs").json()] == ["RET-2", "RET-1"]


def test_logs_limit_is_validated(client):
    assert client.get("/api/logs?limit=0").status_code == 422
    assert client.get("/api/logs?limit=999").status_code == 422


def test_metrics_computed_from_scores(client):
    m = client.get("/api/metrics").json()
    # threshold 0.5 → flagged ids 1,2,3; churners 1,2,4 → TP=2, FP=1, FN=1, TN=2
    assert (m["true_positives"], m["false_positives"], m["false_negatives"], m["true_negatives"]) == (2, 1, 1, 2)
    assert m["recall"] == pytest.approx(2 / 3, abs=1e-3)
    assert m["assumptions"]["ltv_months"] == 24
