"""Agent orchestration — the war-room supervisor's routing decisions
(agents/supervisor.py). Only the deterministic guards are exercised; the single
LLM branch (evidence-sufficiency) is never hit, so no model/network is needed."""

import asyncio

from langgraph.graph import END

from agents.supervisor import supervisor_node, route_from_supervisor, MAX_STEPS


def _next(state: dict) -> str:
    """The node the supervisor routes to for a given (partial) state."""
    return asyncio.run(supervisor_node(state))["next_agent"]


def test_diagnoses_first_when_risk_pending():
    assert _next({"risk_label": "PENDING"}) == "agent1"


def test_low_risk_is_gated_to_finish():
    assert _next({"risk_label": "LOW"}) == "FINISH"


def test_high_risk_gathers_evidence_when_none_yet():
    assert _next({"risk_label": "HIGH"}) == "agent2"


def test_drafts_offer_once_evidence_regather_is_capped():
    state = {"risk_label": "HIGH", "evidence_report": "found similar churners",
             "agent_path": ["agent2:gather", "agent2:gather"]}   # hit MAX_AGENT2_RUNS
    assert _next(state) == "agent3"


def test_finishes_once_offer_is_drafted():
    assert _next({"risk_label": "HIGH", "evidence_report": "e", "retention_offer": "o"}) == "FINISH"


def test_hard_step_cap_forces_finish():
    assert _next({"risk_label": "HIGH", "agent_path": ["x"] * MAX_STEPS}) == "FINISH"


def test_route_from_supervisor_maps_decision_to_node_or_end():
    assert route_from_supervisor({"next_agent": "FINISH"}) is END
    assert route_from_supervisor({"next_agent": "agent2"}) == "agent2"
