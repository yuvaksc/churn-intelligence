"""churn_evals.evaluators — the metrics every analysis is scored on.

LangSmith evaluator signature: f(outputs) -> {"key", "score", "comment"} (or {"results": [...]}).
Everything is read from the analysis's own outputs; the dataset example is only the row the
experiment is attached to.

RAG experiment (churn-rag-golden)
  retrieval_metrics   recall@k, precision@k, nDCG@k (k = 3, 5, 10), MRR of Agent 2's ranked
                      results against relevance labels computed for this customer
  citation_accuracy   every [doc_id] Agent 2 cites was shown to it
  evidence_judge      LLM-as-judge: faithfulness + answer relevance of Agent 2's report

War-room experiment (churn-warroom)
  routing, policy_before_offer, guardrails, citations_ok   code checks
  warroom_judge       LLM-as-judge: is the whole analysis sensible for this customer? (0-1)
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

KS = (3, 5, 10)


def _judge_llm():
    from churn_agents.llm_config import get_analytical_llm
    return get_analytical_llm()


def _na(key: str, why: str = "no offer for a LOW-risk customer") -> dict:
    return {"key": key, "score": None, "comment": f"n/a: {why}"}


# ── Ranking metrics ──────────────────────────────────────────────────────────

def recall_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    """Share of the reachable relevant docs found in the top k (denominator min(k, |relevant|))."""
    relevant = {d for d, g in qrels.items() if g > 0}
    return len(set(ranked[:k]) & relevant) / min(k, len(relevant)) if relevant else 0.0


def precision_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    return sum(1 for d in ranked[:k] if qrels.get(d, 0) > 0) / k


def mrr(ranked: list[str], qrels: dict[str, int]) -> float:
    return next((1.0 / i for i, d in enumerate(ranked, 1) if qrels.get(d, 0) > 0), 0.0)


def ndcg_at_k(ranked: list[str], qrels: dict[str, int], k: int) -> float:
    def dcg(gains):
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = dcg(sorted(qrels.values(), reverse=True)[:k])
    return dcg([qrels.get(d, 0) for d in ranked[:k]]) / ideal if ideal else 0.0


# ── RAG experiment ───────────────────────────────────────────────────────────

def retrieval_metrics(outputs: dict) -> dict:
    ranked, qrels = outputs.get("retrieved_ids") or [], outputs.get("qrels") or {}
    results = [{"key": "mrr", "score": mrr(ranked, qrels)}]
    for k in KS:
        results += [{"key": f"recall@{k}", "score": recall_at_k(ranked, qrels, k)},
                    {"key": f"precision@{k}", "score": precision_at_k(ranked, qrels, k)},
                    {"key": f"ndcg@{k}", "score": ndcg_at_k(ranked, qrels, k)}]
    return {"results": results}


def citation_accuracy(outputs: dict) -> dict:
    c = outputs.get("evidence_citations") or {}
    return {"key": "citation_accuracy", "score": c.get("citation_accuracy"),
            "comment": f"unknown ids: {c['unknown']}" if c.get("unknown") else "every cited id was shown"}


EVIDENCE_JUDGE = """You are grading an evidence report written by a churn-analysis agent.

CONTEXT — exactly what the agent was given (similar customers with ids, the pre-computed
summary, and the documented reasons with counts):
{context}

REPORT TO GRADE:
{report}

Grade two things, strictly:
1. faithful: 1 if EVERY factual claim in the report (patterns, ranges, reasons, counts, quoted
   reasons, which customers say what, whether this customer matches) is supported by the CONTEXT;
   0 if any claim is unsupported, miscounted or invented. Claiming the evidence is insufficient
   when the context does contain similar customers is NOT faithful.
2. relevant: 1 if the report (a) names the dominant churn pattern, (b) names the most common
   reason — or, when no reason appears more than once, says the reasons vary and quotes examples —
   and (c) says whether THIS customer matches, with a reason; else 0."""


class EvidenceGrade(BaseModel):
    faithful: int = Field(description="1 or 0")
    faithful_reason: str = Field(description="One sentence; name any unsupported claim.")
    relevant: int = Field(description="1 or 0")
    relevant_reason: str = Field(description="One sentence.")


def evidence_judge(outputs: dict) -> dict:
    """The judge sees the exact prompt Agent 2 saw, so it grades against the same evidence."""
    g = _judge_llm().with_structured_output(EvidenceGrade).invoke(
        EVIDENCE_JUDGE.format(context=outputs.get("evidence_prompt", ""), report=outputs.get("evidence_report", "")))
    return {"results": [
        {"key": "faithfulness", "score": int(g.faithful), "comment": g.faithful_reason},
        {"key": "answer_relevance", "score": int(g.relevant), "comment": g.relevant_reason},
    ]}


# ── War-room experiment ──────────────────────────────────────────────────────

def routing(outputs: dict) -> dict:
    """LOW customers stop after Agent 1; HIGH customers get an offer."""
    label, path = outputs.get("risk_label"), outputs.get("agent_path") or []
    ok = (label == "LOW" and "agent2" not in path) or (label == "HIGH" and bool(outputs.get("retention_offer")))
    return {"key": "routing", "score": int(ok), "comment": f"{label}: {' → '.join(path)}"}


def policy_before_offer(outputs: dict) -> dict:
    """Agent 3's LLM called the discount-policy tool itself (not the backstop) before drafting."""
    if not outputs.get("retention_offer"):
        return _na("policy_before_offer")
    used = outputs.get("tools_used") or []
    return {"key": "policy_before_offer", "score": int("retention_policy_check" in used),
            "comment": f"LLM tool calls in order: {used}"}


def guardrails(outputs: dict) -> dict:
    g = outputs.get("guardrails")
    if not g:
        return _na("guardrails")
    policy_ok = bool((g.get("policy_ceiling") or {}).get("ok"))
    ground_ok = bool((g.get("grounding") or {}).get("ok"))
    return {"key": "guardrails", "score": int(bool(g.get("passed"))),
            "comment": (f"policy-ceiling {'ok' if policy_ok else 'FAIL'}, grounding "
                        f"{'ok' if ground_ok else 'FAIL'}, {g.get('pii_masked', 0)} PII masked")}


def citations_ok(outputs: dict) -> dict:
    c = outputs.get("evidence_citations")
    if not outputs.get("evidence_report") or not c:
        return _na("citations_ok")
    return {"key": "citations_ok", "score": int(bool(c.get("ok"))),
            "comment": f"unknown ids: {c.get('unknown')}" if c.get("unknown") else "all citations retrieved"}


WARROOM_JUDGE = """You are a senior churn-retention reviewer scoring ONE end-to-end war-room run for a
telecom customer. Decide whether the whole analysis makes sense for THIS customer.

CUSTOMER:
{customer}

AGENT 1 · RISK DIAGNOSIS
risk verdict: {risk_label}   (HIGH = intervene, LOW = no action needed)
risk summary: {risk_summary}

AGENT 2 · EVIDENCE
{evidence}

AGENT 3 · RETENTION OFFER
{offer}

Give ONE score from 0.0 (incoherent) to 1.0 (fully sensible):
- Agent 1: is the HIGH/LOW call and its explanation sensible for this contract, tenure and charge?
- Agent 2: are the similar churners and reasons relevant to this customer? (n/a if LOW)
- Agent 3: does the offer target this customer's churn drivers, stay within the discount
  ceiling, and read as a sound retention play? (n/a if LOW)
A correct LOW-risk run stops after Agent 1: score whether that no-action call is justified.
Your reasoning must touch each agent that ran, in 2-4 sentences."""

_SNAPSHOT = ["Contract", "Tenure Months", "Monthly Charges", "Internet Service",
             "Services Count", "Payment Method", "Online Security", "Tech Support"]


class WarRoomGrade(BaseModel):
    score: float = Field(description="0.0 to 1.0")
    reasoning: str = Field(description="2-4 sentences, touching each agent that ran")


def warroom_judge(outputs: dict) -> dict:
    raw = outputs.get("customer_raw") or {}
    customer = f"US state: {outputs.get('customer_state', 'DEFAULT')} | " + " | ".join(
        f"{k}: {raw[k]}" for k in _SNAPSHOT if k in raw)
    profiles = "\n".join(f"  - {(p.get('document') or '')[:160]}" for p in (outputs.get("similar_profiles") or [])[:5])
    evidence = (f"report: {outputs['evidence_report']}\nsimilar churners:\n{profiles}"
                if outputs.get("evidence_report") else "(none: LOW risk, no evidence gathered)")
    policy = outputs.get("policy") or {}
    offer = (f"offer: {outputs['retention_offer']}\npolicy ceiling: max {policy.get('max_discount_pct')}% / "
             f"${policy.get('max_discount_dollars')}" if outputs.get("retention_offer")
             else "(none: LOW risk, no offer drafted)")
    g = _judge_llm().with_structured_output(WarRoomGrade).invoke(WARROOM_JUDGE.format(
        customer=customer, risk_label=outputs.get("risk_label", "?"),
        risk_summary=outputs.get("risk_summary", "(none)"), evidence=evidence, offer=offer))
    return {"key": "warroom_quality", "score": max(0.0, min(1.0, float(g.score))), "comment": g.reasoning}


RAG_EVALUATORS = [retrieval_metrics, citation_accuracy, evidence_judge]
WARROOM_EVALUATORS = [routing, policy_before_offer, guardrails, citations_ok, warroom_judge]
