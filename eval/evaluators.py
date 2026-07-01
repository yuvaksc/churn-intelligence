"""eval/evaluators.py — LangSmith evaluators (deterministic + Groq LLM-judge).

Each evaluator takes (run, example) and returns {"key", "score", ["reason"]}.
run.outputs is whatever eval.run_eval.target() returned for one example.
"""

import re

from pydantic import BaseModel, Field

from agents.llm_config import get_analytical_llm
from guardrails.policy import check_policy_ceiling

_CITE = re.compile(r"(customer\s*#?\d+|#\d+)", re.I)


# ── Deterministic ─────────────────────────────────────────────────────────────

def policy_ceiling_ok(run, example):
    out = run.outputs or {}
    res = check_policy_ceiling(out.get("retention_offer", ""), out.get("policy", {}))
    return {"key": "policy_ceiling_ok", "score": 1.0 if res["ok"] else 0.0}


def citation_present(run, example):
    out = run.outputs or {}
    ev = out.get("evidence_report", "")
    # LOW-risk runs produce no evidence report → not applicable → pass
    has = bool(_CITE.search(ev)) if ev else True
    return {"key": "citation_present", "score": 1.0 if has else 0.0}


def tools_used_count(run, example):
    out = run.outputs or {}
    return {"key": "tools_used_count", "score": float(len(out.get("tools_used", []) or []))}


def latency_seconds(run, example):
    out = run.outputs or {}
    return {"key": "latency_seconds", "score": float(out.get("latency_s", 0.0) or 0.0)}


# ── LLM-as-judge (Groq) ───────────────────────────────────────────────────────

class _Score(BaseModel):
    score:  float = Field(description="quality from 0.0 (bad) to 1.0 (perfect)")
    reason: str   = Field(description="one short clause justifying the score")


_JUDGE = """You are a strict evaluator. Score from 0.0 to 1.0.

CRITERION: {criterion}

{context}

Output a score (0.0-1.0) and a one-clause reason."""


def _judge(criterion: str, context: str) -> dict:
    if not context.strip():
        return {"score": None, "reason": "not applicable (no content)"}
    try:
        r = get_analytical_llm().with_structured_output(_Score).invoke(
            _JUDGE.format(criterion=criterion, context=context))
        return {"score": max(0.0, min(1.0, float(r.score))), "reason": r.reason}
    except Exception as exc:
        return {"score": None, "reason": f"judge error: {exc}"}


def faithfulness(run, example):
    out = run.outputs or {}
    ev  = out.get("evidence_report", "")
    ctx = f"RETRIEVED EVIDENCE:\n{out.get('evidence_context', '')}\n\nEVIDENCE REPORT:\n{ev}" if ev else ""
    return {"key": "faithfulness",
            **_judge("Is the evidence report fully supported by the retrieved evidence (no invented stats or customers)?", ctx)}


def groundedness(run, example):
    out   = run.outputs or {}
    offer = out.get("retention_offer", "")
    ctx   = (f"POLICY:\n{out.get('policy', {})}\n\nEVIDENCE:\n{out.get('evidence_report', '')}\n\nOFFER:\n{offer}"
             if offer else "")
    return {"key": "groundedness",
            **_judge("Is the retention offer grounded in the policy + evidence and within the discount ceiling?", ctx)}


def answer_relevance(run, example):
    out   = run.outputs or {}
    offer = out.get("retention_offer", "")
    ctx   = f"RISK SUMMARY:\n{out.get('risk_summary', '')}\n\nOFFER:\n{offer}" if offer else ""
    return {"key": "answer_relevance",
            **_judge("Does the offer directly address this customer's specific churn-risk drivers?", ctx)}


ALL_EVALUATORS = [
    policy_ceiling_ok, citation_present, tools_used_count, latency_seconds,
    faithfulness, groundedness, answer_relevance,
]
