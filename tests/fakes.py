"""Test doubles for the interfaces in churn_ml/churn_rag/churn_data/churn_agents.

No network, no model files, no database: the war-room graph and the API can be
built entirely from these.
"""

import json
from contextlib import asynccontextmanager

from langchain_core.messages import AIMessage


# ── churn_ml.predictor.Predictor ─────────────────────────────────────────────────
class FakePredictor:
    model_version = "fake:1"

    def __init__(self, risk_score: float, threshold: float = 0.5):
        self.risk_score, self.threshold, self.calls = risk_score, threshold, 0

    def predict(self, customer_raw: dict) -> dict:
        self.calls += 1
        return {
            "risk_score": self.risk_score,
            "risk_label": "HIGH" if self.risk_score >= self.threshold else "LOW",
            "threshold": self.threshold,
            "shap_drivers": [{"feature": "Contract", "shap_value": 0.8,
                              "raw_value": 1.0, "direction": "increases churn risk"}],
        }


# ── churn_rag.ports.Retriever ────────────────────────────────────────────────
class FakeRetriever:
    def __init__(self, docs: list[dict] | None = None):
        self.docs = docs if docs is not None else [
            {"doc_id": "p1", "document": "Month-to-month | Fiber optic | tenure 3",
             "metadata": {"churn_label": 1, "churn_reason": "Competitor offered more data"},
             "similarity": 0.91},
            {"doc_id": "p2", "document": "Month-to-month | Fiber optic | tenure 5",
             "metadata": {"churn_label": 1, "churn_reason": "Competitor offered more data"},
             "similarity": 0.88},
        ]
        self.calls: list[str] = []

    def retrieve(self, query, k, *, churners_only=True, filters=None, near=None):
        self.calls.append(query)
        return self.docs[:k]


# ── LLMs ─────────────────────────────────────────────────────────────────────
class _Structured:
    def __init__(self, schema, values):
        self.schema, self.values = schema, values

    async def ainvoke(self, _prompt):
        return self.schema(**self.values)


class ScriptedLLM:
    """Returns `text` for plain calls, `route` for the supervisor's structured call,
    and pops scripted AIMessages (e.g. tool calls) first when a script is given."""

    def __init__(self, text="A short analysis.", route="agent3", script=None):
        self.text, self.route = text, route
        self.script = list(script or [])
        self.prompts: list = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        if self.script:
            return self.script.pop(0)
        return AIMessage(content=self.text)

    def with_structured_output(self, schema):
        return _Structured(schema, {"next": self.route, "reason": "scripted"})

    def bind_tools(self, specs):
        self.bound_specs = specs
        return self


def tool_call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


# ── churn_agents.tools.gateway.ToolGateway ──────────────────────────────────
class FakeTools:
    RESPONSES = {
        "retention_policy_check": {"max_discount_pct": 15, "max_discount_usd": 20.0},
        "get_competitor_insights": {"competitor": "X", "offer": "more data"},
        "log_retention_action": {"log_id": "RET-TEST-1"},
    }

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def list_specs(self):
        return [{"type": "function", "function": {"name": n, "description": "", "parameters": {}}}
                for n in self.RESPONSES]

    async def call(self, name, args):
        self.calls.append((name, args))
        return json.dumps(self.RESPONSES.get(name, {}))

    @property
    def names(self) -> list[str]:
        return [n for n, _ in self.calls]

    def factory(self):
        @asynccontextmanager
        async def _open():
            yield self
        return _open


def initial_state(customer_raw: dict | None = None) -> dict:
    return {
        "customer_id": "TEST-1", "customer_state": "CA",
        "customer_raw": customer_raw or {"Contract": "Month-to-month", "Monthly Charges": 80.0,
                                         "Tenure Months": 3, "Internet Service": "Fiber optic"},
        "risk_score": 0.0, "risk_label": "PENDING", "shap_drivers": [], "risk_summary": "",
        "similar_profiles": [], "churn_reasons": [], "evidence_report": "",
        "policy": {}, "competitor_intel": {}, "retention_offer": "", "crm_log_id": "",
        "crm_logged": False, "tools_used": [], "agent_path": [],
    }
