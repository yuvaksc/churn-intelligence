"""eval/run_eval.py — run the war room over a LangSmith dataset and score it.

Creates (once) a LangSmith dataset of sample customer inputs, runs the full war
room on each as the target, applies the deterministic + LLM-judge evaluators, and
logs a LangSmith experiment. Tracing of the internal LLM/agent calls is automatic
when LANGSMITH_TRACING=true.

Run inside the api container (deps + LANGSMITH_* env + MCP network live there):
    docker compose exec api python -m eval.run_eval
"""

import os
import sys
import time
import asyncio

try:
    from langsmith import Client, evaluate
except ImportError:                       # older layout
    from langsmith import Client
    from langsmith.evaluation import evaluate

from agents.graph import war_room_graph
from db import customers as db_customers
from eval.evaluators import ALL_EVALUATORS

DATASET = "churn-eval"

# Sample agent inputs — a mix of HIGH-risk customers (full war room) and a
# LOW-risk one (risk-gated, no offer) so the evaluators see both paths.
_EXAMPLES = [
    {"customer_id": 1818, "customer_state": "California"},
    {"customer_id": 641,  "customer_state": "Texas"},
    {"customer_id": 5716, "customer_state": "DEFAULT"},   # LOW risk
]


def _ensure_dataset(client: "Client"):
    try:
        ds = client.read_dataset(dataset_name=DATASET)
        print(f"dataset '{DATASET}' already exists")
        return ds
    except Exception:
        ds = client.create_dataset(DATASET, description="Churn war-room eval — sample customer inputs")
        client.create_examples(inputs=_EXAMPLES, dataset_id=ds.id)
        print(f"created dataset '{DATASET}' with {len(_EXAMPLES)} examples")
        return ds


def _evidence_context(state: dict) -> str:
    profs   = "\n".join((p.get("document") or "")[:160] for p in state.get("similar_profiles", [])[:5])
    reasons = ", ".join(r.get("reason", "") for r in state.get("churn_reasons", [])[:8])
    return f"SIMILAR CHURNERS:\n{profs}\n\nCHURN REASONS: {reasons}"


def target(inputs: dict) -> dict:
    """Run the full war room on one example and return the fields the evaluators need."""
    cid = int(inputs["customer_id"])
    raw = db_customers.get_customer_features(cid)
    if raw is None:
        return {"error": f"customer {cid} not found"}
    initial = {
        "customer_id": f"TEST-{cid}", "customer_raw": raw,
        "customer_state": inputs.get("customer_state", "DEFAULT"),
        "risk_score": 0.0, "risk_label": "PENDING", "shap_drivers": [], "risk_summary": "",
        "similar_profiles": [], "churn_reasons": [], "evidence_report": "",
        "policy": {}, "competitor_intel": {}, "retention_offer": "", "crm_log_id": "",
        "crm_logged": False, "tools_used": [], "messages": [],
    }
    t0 = time.perf_counter()
    state = asyncio.run(war_room_graph.ainvoke(initial))
    return {
        "risk_label":       state.get("risk_label"),
        "risk_summary":     state.get("risk_summary", ""),
        "evidence_report":  state.get("evidence_report", ""),
        "evidence_context": _evidence_context(state),
        "retention_offer":  state.get("retention_offer", ""),
        "policy":           state.get("policy", {}),
        "tools_used":       state.get("tools_used", []),
        "agent_path":       state.get("agent_path", []),
        "latency_s":        round(time.perf_counter() - t0, 2),
    }


def main() -> None:
    if not os.getenv("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY not set — aborting.")
        sys.exit(1)

    client = Client()
    _ensure_dataset(client)

    print(f"running experiment over '{DATASET}' (this runs the full war room per example)...")
    evaluate(
        target,
        data=DATASET,
        evaluators=ALL_EVALUATORS,
        experiment_prefix="churn-warroom",
        client=client,
        max_concurrency=1,
    )
    print(f"\nDone. View results in LangSmith → Datasets & Experiments → '{DATASET}'.")


if __name__ == "__main__":
    main()
