"""
churn_agents/agent2_researcher.py — Evidence Researcher (Agent 2).  [async]

Retrieves similar historical churners (pgvector + full-text hybrid search, pre-filtered
on the customer's contract and internet type, reranked by a cross-encoder), ranks
their documented churn reasons, and asks the LLM for a cited evidence report. A
guardrail then checks that every cited [doc_id] was actually retrieved.
"""

import asyncio

from langsmith import traceable

from churn_agents.deps import WarRoomDeps
from churn_agents import prompts
from churn_agents.guardrails.citations import check_citations
from churn_agents.state import WarRoomState

_DISPLAY_K   = 5    # similar churners shown to the LLM / UI
_REASON_POOL = 12   # wider pool retrieved only for the reason-frequency signal


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
        reason  = (p["metadata"].get("churn_reason") or "").strip()
        doc     = p["document"][:160].rstrip("|").strip()
        tail    = f'  → left: "{reason}"' if reason else ""
        ref = p.get("doc_id") or f"profile_{i}"
        lines.append(f"  [{ref}] {churned} (sim={p['similarity']:.2f})  {doc}...{tail}")
    return "\n".join(lines)


def _aggregate_reasons(profiles: list) -> list[dict]:
    """Frequency-rank the documented churn reasons of the retrieved churners. Each entry
    keeps the same shape the rest of the app expects: reason / metadata.count / similarity."""
    agg: dict[str, dict] = {}
    for p in profiles:
        reason = ((p.get("metadata") or {}).get("churn_reason") or "").strip()
        if not reason:
            continue
        e = agg.setdefault(reason, {"reason": reason, "count": 0, "similarity": 0.0})
        e["count"]     += 1
        e["similarity"] = max(e["similarity"], float(p.get("similarity", 0.0)))
    ranked = sorted(agg.values(), key=lambda x: (-x["count"], -x["similarity"]))
    return [
        {"reason": e["reason"], "metadata": {"count": e["count"]}, "similarity": round(e["similarity"], 4)}
        for e in ranked
    ]


def _format_reasons(reasons: list) -> str:
    return "\n".join(f"  • {r['reason']}  (×{r['metadata']['count']})" for r in reasons[:3])


def _profile_stats(profiles: list) -> str:
    """Exact ranges and counts over the profiles shown to the LLM, computed in code:
    LLMs are unreliable at min/max arithmetic, so the prompt gives them the answer."""
    meta = [p.get("metadata") or {} for p in profiles]
    tenures = [int(m["tenure_months"]) for m in meta if m.get("tenure_months") is not None]
    charges = [float(m["monthly_charges"]) for m in meta if m.get("monthly_charges") is not None]

    def counts(key: str) -> str:
        c: dict[str, int] = {}
        for m in meta:
            if m.get(key):
                c[m[key]] = c.get(m[key], 0) + 1
        return ", ".join(f"{k} ×{n}" for k, n in sorted(c.items(), key=lambda kv: -kv[1])) or "unknown"

    parts = [f"{len(profiles)} customers"]
    if tenures:
        parts.append(f"tenure {min(tenures)} to {max(tenures)} months")
    if charges:
        parts.append(f"monthly charges ${min(charges):.2f} to ${max(charges):.2f}")
    parts.append(f"contracts: {counts('contract')}")
    parts.append(f"internet: {counts('internet_service')}")
    return "  " + " | ".join(parts)


def _as_documents(profiles: list) -> dict:
    """Retriever-span output in LangSmith's document shape (shown as a ranked list)."""
    return {"documents": [{"page_content": p.get("document", ""),
                           "metadata": {"doc_id": p.get("doc_id"), "similarity": p.get("similarity"),
                                        **(p.get("metadata") or {})}} for p in profiles]}


@traceable(name="hybrid_search", run_type="retriever",
           process_inputs=lambda i: {k: i.get(k) for k in ("query", "k", "filters", "near")},
           process_outputs=_as_documents)
def retrieve_similar_churners(retriever, query: str, k: int, filters: dict | None,
                              near: dict | None = None) -> list:
    """Dense + keyword + structured search, RRF-fused, cross-encoder reranked (one trace span)."""
    return retriever.retrieve(query, k, churners_only=True, filters=filters, near=near)


def _litm_order(items: list) -> list:
    """Lost-in-the-middle packing: place the strongest items at the start AND end
    of the list and the weakest in the middle (LLMs attend least to the middle of
    a long context). Input must be ranked best-first."""
    head: list = []
    tail: list = []
    for i, item in enumerate(items):
        (head if i % 2 == 0 else tail).append(item)
    return head + tail[::-1]


def build_evidence_prompt(risk_summary: str, snapshot: str, similar_profiles: list) -> tuple[str, list[str]]:
    """Agent 2's user message and the ids it shows, in display order."""
    ordered = _litm_order(similar_profiles)
    user = prompts.EVIDENCE_USER.format(
        risk_summary=risk_summary,
        customer_snapshot=snapshot,
        similar_profiles=_format_profiles(ordered) or "  (none found)",
        profile_stats=_profile_stats(similar_profiles) if similar_profiles else "  (none)",
        # Counted over the customers SHOWN, so every count is verifiable from the prompt.
        # (state["churn_reasons"] keeps the wider pool for the UI.)
        churn_reasons=_format_reasons(_aggregate_reasons(similar_profiles)) or "  (none documented)",
    )
    return user, [p.get("doc_id") or f"profile_{i}" for i, p in enumerate(ordered, 1)]


async def agent2_node(state: WarRoomState, deps: WarRoomDeps) -> dict:
    print("\n" + "─" * 52)
    print("  [Agent 2 — Evidence Researcher]")
    print("─" * 52)

    snapshot = _customer_snapshot(state["customer_raw"])

    # One hybrid retrieval; each churner carries its own documented reason. Pull a wider pool
    # for the reason-frequency signal, show the top few. Blocking I/O + CPU → thread.
    #  - The query is the customer's profile ONLY. Adding the LLM's risk summary made the
    #    prose dominate the search and cut nDCG@5 from 0.31 to 0.10 (measured).
    #  - Hard pre-filter on contract + internet type; soft structured signal on tenure,
    #    charge and payment method, which text search cannot compare (README).
    raw     = state["customer_raw"]
    filters = {k: v for k, v in {"contract": raw.get("Contract"),
                                 "internet_service": raw.get("Internet Service")}.items() if v}
    near    = {"tenure_months": int(raw.get("Tenure Months", 0) or 0),
               "monthly_charges": float(raw.get("Monthly Charges", 0) or 0),
               "payment_method": raw.get("Payment Method")}
    pool             = await asyncio.to_thread(
        retrieve_similar_churners, deps.retriever, snapshot, _REASON_POOL, filters or None, near)
    similar_profiles = pool[:_DISPLAY_K]
    churn_reasons    = _aggregate_reasons(pool)

    print(f"  Similar churners retrieved:  {len(pool)} (showing {len(similar_profiles)})")
    print(f"  Distinct churn reasons:      {len(churn_reasons)}")
    if similar_profiles:
        top = similar_profiles[0]
        print(f"  Most similar: similarity={top['similarity']:.2f}  "
              f"{top['document'][:80]}...")

    user, shown_ids = build_evidence_prompt(state["risk_summary"], snapshot, similar_profiles)
    response        = await deps.analytical_llm().ainvoke(prompts.messages(prompts.EVIDENCE_SYSTEM, user))
    evidence_report = response.content.strip()

    print(f"  Evidence:     {evidence_report[:100]}...")

    # Citation guardrail: every [doc_id] must be one of the profiles shown to the LLM.
    citations = check_citations(evidence_report, shown_ids)
    if not citations["ok"]:
        print(f"  [Guardrail] citation check failed: unknown={citations['unknown']}")

    return {
        "similar_profiles": similar_profiles,
        "churn_reasons":    churn_reasons,
        "evidence_report":  evidence_report,
        "evidence_citations": citations,
        # for the evals: the full ranked result and the exact prompt the LLM saw
        "retrieved_ids":    [p.get("doc_id") for p in pool],
        "evidence_prompt":  user,
    }