"""
agents/agent2_researcher.py — Evidence Researcher (Agent 2).  [async]

Changes from sync version:
  - async def agent2_node
  - ChromaDB queries wrapped in asyncio.to_thread (blocking I/O)
  - llm.invoke() → await llm.ainvoke()
"""

import asyncio

from agents.state import WarRoomState
from agents.llm_config import get_analytical_llm
from rag.retriever import query_similar_profiles, query_churn_reasons


_EVIDENCE_PROMPT = """\
You are a churn evidence researcher. The Diagnostic Agent has flagged a high-risk customer.

RISK SUMMARY FROM AGENT 1:
{risk_summary}

CURRENT CUSTOMER KEY FEATURES:
{customer_snapshot}

TOP 5 MOST SIMILAR HISTORICAL CUSTOMERS (churners only):
{similar_profiles}

TOP CHURN REASONS FROM SIMILAR CUSTOMERS:
{churn_reasons}

Write a 3-sentence Evidence Report for the retention specialist:
  Sentence 1: What is the dominant churn pattern for customers with this profile?
  Sentence 2: What is the single most common reason these customers left?
              Quote the reason verbatim if it appears multiple times.
  Sentence 3: Does this specific customer match that pattern?
              Yes/No + one specific reason why or why not.

Rules:
  - Use ONLY the evidence provided — do not invent statistics or customers
  - Cite specifics: refer to the similar customers by their list number (e.g. "customer #1")
    and quote churn reasons verbatim
  - If the retrieved evidence is weak or does not match this customer, say so explicitly —
    write "Insufficient historical evidence to establish a pattern" rather than guessing
  - No bullet points — flowing sentences only
  - Do NOT recommend any offer
"""

_SNAPSHOT_FIELDS = [
    "Contract", "Internet Service", "Tenure Months", "Monthly Charges",
    "Services Count", "High Risk Flag", "Online Security", "Tech Support",
    "Payment Method", "Paperless Billing", "Partner", "Dependents",
]


def _customer_snapshot(customer_raw: dict) -> str:
    return " | ".join(
        f"{k}: {customer_raw[k]}"
        for k in _SNAPSHOT_FIELDS
        if k in customer_raw
    )


def _format_profiles(profiles: list) -> str:
    lines = []
    for i, p in enumerate(profiles, 1):
        churned = "CHURNED" if p["metadata"]["churn_label"] == 1 else "STAYED"
        doc     = p["document"][:160].rstrip("|").strip()
        lines.append(f"  {i}. [{churned}] (sim={p['similarity']:.2f})  {doc}...")
    return "\n".join(lines)


def _format_reasons(reasons: list) -> str:
    counts: dict[str, int] = {}
    for r in reasons:
        counts[r["reason"]] = counts.get(r["reason"], 0) + 1
    top3 = sorted(counts.items(), key=lambda x: -x[1])[:3]
    return "\n".join(f"  • {reason}  (×{count})" for reason, count in top3)


def _litm_order(items: list) -> list:
    """Lost-in-the-middle packing: place the strongest items at the start AND end
    of the list and the weakest in the middle (LLMs attend least to the middle of
    a long context). Input must be ranked best-first."""
    head, tail = [], []
    for i, item in enumerate(items):
        (head if i % 2 == 0 else tail).append(item)
    return head + tail[::-1]


async def agent2_node(state: WarRoomState) -> dict:
    print("\n" + "─" * 52)
    print("  [Agent 2 — Evidence Researcher]")
    print("─" * 52)

    snapshot   = _customer_snapshot(state["customer_raw"])
    query_text = f"{state['risk_summary']} | {snapshot}"

    # Hybrid retrieval (dense + BM25 + RRF + cross-encoder rerank) is blocking
    # CPU work → offload to a thread so the event loop stays free.
    similar_profiles = await asyncio.to_thread(query_similar_profiles, query_text, 5, True)
    churn_reasons    = await asyncio.to_thread(query_churn_reasons, query_text, 10)

    print(f"  Similar churner profiles retrieved:  {len(similar_profiles)}")
    print(f"  Churn reasons retrieved:             {len(churn_reasons)}")
    if similar_profiles:
        top = similar_profiles[0]
        print(f"  Most similar: similarity={top['similarity']:.2f}  "
              f"{top['document'][:80]}...")

    prompt = _EVIDENCE_PROMPT.format(
        risk_summary=state["risk_summary"],
        customer_snapshot=snapshot,
        similar_profiles=_format_profiles(_litm_order(similar_profiles)),
        churn_reasons=_format_reasons(churn_reasons),
    )
    response        = await get_analytical_llm().ainvoke(prompt)
    evidence_report = response.content.strip()

    print(f"  Evidence:     {evidence_report[:100]}...")

    return {
        "similar_profiles": similar_profiles,
        "churn_reasons":    churn_reasons,
        "evidence_report":  evidence_report,
    }