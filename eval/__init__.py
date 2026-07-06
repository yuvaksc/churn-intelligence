"""eval/ — live war-room evaluation as a real experiment run on the churn-eval dataset.

The war room is run *inside* an experiment (`api/routes/ws.py` attaches the callbacks
from `experiment_callbacks(...)`), so the actual graph execution — with its full
agent1/2/3 waterfall — IS the experiment run under `churn-eval` (linked to the golden
of the same risk label as its reference archetype). After it finishes, `score_run(...)`
attaches the metrics as feedback on that run:

  1. output_completeness — did the run produce all the JSON elements the reference
     archetype has, for its risk label? (a LOW run is correctly gated after Agent 1
     and lacks the Agent 2 / Agent 3 blocks a HIGH run carries). Score = fraction of
     the archetype's populated fields the run also populated.
  2/3. warroom_quality — one holistic Groq judge: an overall score for the agents'
     output, and the reasoning behind it (the feedback comment).
  4. guardrails — did the output guardrails (PII / policy-ceiling / grounding) all
     pass, with a short reasoning (offer runs only; N/A when there is no offer).

The golden examples are per-label ARCHETYPES of the expected output (fetched from the
`churn-eval` dataset by risk label, cached per process), NOT a per-customer answer key.
"""

import os
import uuid
import datetime

from pydantic import BaseModel, Field

from agents.llm_config import get_analytical_llm

DATASET           = "churn-eval"
EXPERIMENT_PREFIX = "churn-warroom"
COMPLETENESS_KEY  = "output_completeness"
QUALITY_KEY       = "warroom_quality"
GUARDRAILS_KEY    = "guardrails"

_SNAPSHOT_FIELDS = [
    "Contract", "Tenure Months", "Monthly Charges", "Internet Service",
    "Services Count", "Payment Method", "Online Security", "Tech Support",
]


def _tracing_on() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() in ("1", "true", "yes")


def _agent(outputs, name: str) -> dict:
    """The WS block for one agent (empty dict if that agent never completed)."""
    return (outputs or {}).get(name) or {}


# ── LangSmith client + dataset handles (cached singletons) ────────────────────────

_client_singleton = None
_dataset_id_cache = None
_reference_cache: dict | None = None


def _client():
    global _client_singleton
    if _client_singleton is None:
        from langsmith import Client
        _client_singleton = Client()
    return _client_singleton


def _dataset_id():
    global _dataset_id_cache
    if _dataset_id_cache is None:
        _dataset_id_cache = _client().read_dataset(dataset_name=DATASET).id
    return _dataset_id_cache


def _references_by_label() -> dict:
    """One golden EXAMPLE per risk label from `churn-eval` — the archetype of the
    expected output for that label (carries both `.outputs` and `.id`). Cached."""
    global _reference_cache
    if _reference_cache is None:
        cache: dict = {}
        for ex in _client().list_examples(dataset_name=DATASET):
            label = ((ex.outputs or {}).get("agent1") or {}).get("risk_label")
            if label and label not in cache:
                cache[label] = ex
        _reference_cache = cache
    return _reference_cache


# ── Metric 1: structural completeness vs the same-label golden archetype ──────────

_COMPLETENESS_BLOCKS = ("agent1", "agent2", "agent3")
_EMPTY = (None, "", [], {})


def _expected_fields(reference: dict) -> list[tuple[str, str]]:
    """(agent, field) pairs the archetype populates — the JSON elements a run of this
    label is expected to produce. `guardrails` is a computed report, not an output."""
    expected = []
    for a in _COMPLETENESS_BLOCKS:
        for k, v in (reference.get(a) or {}).items():
            if k != "guardrails" and v not in _EMPTY:
                expected.append((a, k))
    return expected


def _completeness(outputs: dict, reference: dict, label: str) -> tuple[float, str]:
    """Fraction of the archetype's populated JSON elements the run also populated."""
    expected = _expected_fields(reference)
    if not expected:
        raise ValueError("no reference fields for this label")
    out = outputs or {}
    present, missing = 0, []
    for a, k in expected:
        if (out.get(a) or {}).get(k) not in _EMPTY:
            present += 1
        else:
            missing.append(f"{a}.{k}")
    score = present / len(expected)
    if missing:
        comment = f"{present}/{len(expected)} expected JSON elements present vs the {label} archetype; missing: {', '.join(missing[:12])}"
    else:
        comment = f"all {len(expected)} expected JSON elements present — matches the {label} archetype"
    return score, comment


# ── Metric 4: did the output guardrails all pass? ─────────────────────────────────

def _guardrails_check(outputs: dict) -> tuple[float, str] | None:
    """1.0 if the offer's guardrails (PII / policy-ceiling / grounding) all passed,
    else 0.0, with a short reasoning. None when there is no offer (LOW risk)."""
    g = _agent(outputs, "agent3").get("guardrails")
    if not g:
        return None
    policy_ok = bool((g.get("policy_ceiling") or {}).get("ok"))
    ground_ok = bool((g.get("grounding") or {}).get("ok"))
    passed    = bool(g.get("passed"))
    reasoning = (
        f"policy-ceiling {'ok' if policy_ok else 'FAIL'}, "
        f"grounding {'ok' if ground_ok else 'FAIL'}, "
        f"{g.get('pii_masked', 0)} PII masked"
    )
    verdict = "all guardrails passed" if passed else "guardrails FAILED"
    return (1.0 if passed else 0.0), f"{verdict} — {reasoning}"


# ── Metric 2/3: one holistic Groq judge over the run's reasoning ──────────────────

def _customer_snapshot(inputs: dict) -> str:
    raw = (inputs or {}).get("customer_raw", {}) or {}
    fields = " | ".join(f"{k}: {raw[k]}" for k in _SNAPSHOT_FIELDS if k in raw)
    return f"US state: {(inputs or {}).get('customer_state', 'DEFAULT')} | {fields}"


def _evidence_block(agent2: dict) -> str:
    report  = agent2.get("evidence_report", "")
    profs   = "\n".join(f"  - {(p.get('document') or '')[:160]}" for p in (agent2.get("similar_profiles") or [])[:5])
    reasons = ", ".join(r.get("reason", "") for r in (agent2.get("churn_reasons") or [])[:6])
    if not (report or profs):
        return "(none — LOW risk, no evidence gathered)"
    return f"evidence_report: {report}\nsimilar churners:\n{profs}\nchurn reasons: {reasons}"


def _offer_block(agent3: dict) -> str:
    offer = agent3.get("retention_offer", "")
    if not offer:
        return "(none — LOW risk, no offer drafted)"
    policy  = agent3.get("policy", {}) or {}
    ceiling = f"max {policy.get('max_discount_pct')}% / ${policy.get('max_discount_dollars')}"
    return f"offer: {offer}\npolicy ceiling: {ceiling}\ntools used: {agent3.get('tools_used', [])}"


class _Assessment(BaseModel):
    score:     float = Field(description="overall quality 0.0 (incoherent / irrelevant) to 1.0 (fully sensible)")
    reasoning: str   = Field(description="2-4 sentences assessing each agent that ran: the risk call, the evidence relevance, and the offer quality")


_PROMPT = """You are a senior churn-retention reviewer scoring ONE end-to-end war-room run for a telecom customer. Decide whether the whole analysis makes sense for THIS specific customer, and give a single overall score with reasoning.

CUSTOMER (the input):
{customer}

--- AGENT 1 · RISK DIAGNOSIS ---
risk verdict: {risk_label}   (HIGH = intervene, LOW = no action needed)
risk summary: {risk_summary}

--- AGENT 2 · EVIDENCE (similar past churners + reasons) ---
{evidence}

--- AGENT 3 · RETENTION OFFER ---
{offer}

Assess holistically and give ONE score (0.0-1.0):
- Agent 1: given this customer's contract, tenure, charges and state, is the HIGH/LOW call and its reasoning sensible?
- Agent 2: are the retrieved similar churners and reasons genuinely relevant to this customer? (n/a if LOW risk)
- Agent 3: does the offer target this customer's actual churn drivers, stay within the discount ceiling, and read as a sound retention play? (n/a if LOW risk)

A correct LOW-risk run stops after Agent 1 with no evidence or offer — score whether that no-action call is justified.
Your reasoning must explicitly touch each agent that ran."""


def judge(inputs: dict, outputs: dict) -> tuple[float, str]:
    """Holistic Groq judge over ONE finished war-room run (reference-free).
    Returns (overall score 0.0-1.0, reasoning behind the score)."""
    a1, a2, a3 = _agent(outputs, "agent1"), _agent(outputs, "agent2"), _agent(outputs, "agent3")
    prompt = _PROMPT.format(
        customer=_customer_snapshot(inputs),
        risk_label=a1.get("risk_label", "?"),
        risk_summary=a1.get("risk_summary", "(none)"),
        evidence=_evidence_block(a2),
        offer=_offer_block(a3),
    )
    r = get_analytical_llm().with_structured_output(_Assessment).invoke(prompt)
    return max(0.0, min(1.0, float(r.score))), r.reasoning


# ── Live API used by the WebSocket ────────────────────────────────────────────────

def experiment_callbacks(risk_label: str, customer_id) -> list:
    """Callbacks that make the war-room run an experiment run under `churn-eval`,
    linked to the same-label golden archetype (so the run's real waterfall shows in
    Datasets & Experiments). Returns [] when tracing is off or the dataset is
    unreachable — the run then just traces normally. Best-effort (never raises).

    ONE experiment per analysis (named per customer + timestamp) so every run is its
    own entry, instead of same-label runs collapsing into repetitions of the shared
    archetype example. Do the network setup in a thread (the caller wraps it)."""
    if not _tracing_on():
        return []
    try:
        from langsmith.utils import LangSmithConflictError
        try:
            from langchain_core.tracers import LangChainTracer
        except Exception:
            from langchain_core.tracers.langchain import LangChainTracer

        client = _client()
        ref    = _references_by_label().get(risk_label)
        stamp  = datetime.datetime.now().strftime("%H%M%S")
        exp    = f"{EXPERIMENT_PREFIX}-{customer_id}-{stamp}-{uuid.uuid4().hex[:6]}"
        try:
            client.create_project(exp, reference_dataset_id=_dataset_id())
        except LangSmithConflictError:                   # unique name, but stay safe on collision
            pass
        return [LangChainTracer(project_name=exp, example_id=(ref.id if ref else None), client=client)]
    except Exception:
        return []


def score_run(run_id, inputs: dict, outputs: dict) -> dict | None:
    """Attach the four eval metrics as feedback on the experiment run `run_id`:
    output_completeness, warroom_quality (score + reasoning), guardrails. Best-effort
    — never raises, each metric logged independently."""
    if not (run_id and _tracing_on()):
        return None
    try:
        client = _client()
    except Exception:
        return None

    label = _agent(outputs, "agent1").get("risk_label", "")
    try:
        ref = _references_by_label().get(label)
    except Exception:
        ref = None
    results: dict = {}

    # 1 — did we get all the JSON elements the reference archetype has?
    try:
        score, comment = _completeness(outputs, (ref.outputs if ref else {}), label or "?")
        client.create_feedback(run_id, key=COMPLETENESS_KEY, score=score, comment=comment)
        results[COMPLETENESS_KEY] = score
    except Exception:
        pass

    # 2/3 — overall quality score + the reasoning behind it
    try:
        score, reasoning = judge(inputs, outputs)
        client.create_feedback(run_id, key=QUALITY_KEY, score=score, comment=reasoning)
        results[QUALITY_KEY] = score
    except Exception:
        pass

    # 4 — did the output guardrails all pass? (offer runs only)
    try:
        gr = _guardrails_check(outputs)
        if gr is not None:
            score, comment = gr
            client.create_feedback(run_id, key=GUARDRAILS_KEY, score=score, comment=comment)
            results[GUARDRAILS_KEY] = score
    except Exception:
        pass

    try:
        client.flush()
    except Exception:
        pass
    return results or None
