"""churn_evals.live — every analysis becomes a LangSmith experiment.

The API calls `log_experiments` after an analysis has been sent to the UI. It records:

  churn-warroom      experiment "warroom-<customer>-prompt-<v>"  (every analysis)
  churn-rag-golden   experiment "rag-<customer>-prompt-<v>"      (HIGH risk, where Agent 2 ran)

with the analysis as the experiment's output and the metrics in `evaluators.py` as its
scores. Each experiment is attached to a fixed reference example (the same customer always
lands on the same example), so after a prompt change you re-run the same customer and compare
the two experiments side by side. The full waterfall stays in the analysis's own trace
(`trace_id` in each experiment's output and metadata).

RAG relevance labels for the analysed customer are computed from document metadata:
grade 2 = a churner with the same contract, internet type, tenure bucket, charge band AND
payment method; grade 1 = same but a different payment method.
"""

from __future__ import annotations

import os

from churn_evals import evaluators as ev

WARROOM_DATASET = "churn-warroom"
RAG_DATASET = "churn-rag-golden"


def tracing_on() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() in ("1", "true", "yes")


def relevance_labels(customer_raw: dict) -> dict[str, int]:
    from churn_data.db import tx
    from churn_rag.documents import monthly_band, tenure_bucket
    with tx() as c:
        rows = c.execute(
            "select doc_id, payment_method from rag_documents where churn_label = 1 and contract = %s "
            "and internet_service = %s and tenure_bucket = %s and monthly_band = %s",
            [customer_raw.get("Contract"), customer_raw.get("Internet Service"),
             tenure_bucket(int(customer_raw.get("Tenure Months", 0) or 0)),
             monthly_band(float(customer_raw.get("Monthly Charges", 0) or 0))]).fetchall()
    return {r["doc_id"]: 2 if r["payment_method"] == customer_raw.get("Payment Method") else 1 for r in rows}


def _example(client, dataset: str, customer_id: int):
    """Fixed reference example for this customer (same customer → same example)."""
    examples = sorted(client.list_examples(dataset_name=dataset), key=lambda e: str(e.id))
    return examples[customer_id % len(examples)]


def _experiment(client, dataset: str, prefix: str, outputs: dict, evaluators: list, metadata: dict) -> str:
    from langsmith import evaluate
    res = evaluate(lambda _inputs: outputs, data=[_example(client, dataset, metadata["customer_id"])],
                   evaluators=evaluators, experiment_prefix=prefix, metadata=metadata, max_concurrency=0)
    return res.experiment_name


def log_experiments(trace_id, customer_id: int, state: dict) -> list[str]:
    """Record this analysis as experiments on both datasets. Best-effort: never raises."""
    if not tracing_on():
        return []
    from langsmith import Client
    from churn_agents.prompts import PROMPT_VERSION

    client = Client()
    meta = {"customer_id": customer_id, "prompt_version": PROMPT_VERSION, "trace_id": str(trace_id),
            "risk_label": state.get("risk_label")}
    outputs = {**state, "trace_id": str(trace_id)}
    names = []
    try:
        names.append(_experiment(client, WARROOM_DATASET, f"warroom-{customer_id}-prompt-{PROMPT_VERSION}",
                                 outputs, ev.WARROOM_EVALUATORS, meta))
        if state.get("evidence_report"):
            rag_out = {k: outputs.get(k) for k in ("retrieved_ids", "evidence_prompt", "evidence_report",
                                                   "evidence_citations", "similar_profiles", "trace_id")}
            rag_out["qrels"] = relevance_labels(state.get("customer_raw") or {})
            names.append(_experiment(client, RAG_DATASET, f"rag-{customer_id}-prompt-{PROMPT_VERSION}",
                                     rag_out, ev.RAG_EVALUATORS, meta))
    except Exception as exc:
        print(f"  [evals] experiment not recorded: {exc}")
    return names
