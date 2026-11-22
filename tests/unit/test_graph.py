"""War-room graph with every dependency faked: deterministic, no network.

Covers the routing guarantees (low risk skips research + mitigation) and the
tool-use guarantee that a discount is never drafted without the policy check.
"""

import asyncio

from churn_agents.deps import WarRoomDeps
from churn_agents.graph import build_graph
from fakes import FakePredictor, FakeRetriever, FakeTools, ScriptedLLM, initial_state, tool_call

OFFER = "HOOK: Thanks for 3 months.\nOFFER: Take 10% off for a year.\nURGENCY: Reply this week."


def _run(predictor, llm=None, creative=None, tools=None, retriever=None):
    tools, retriever = tools or FakeTools(), retriever or FakeRetriever()
    llm, creative = llm or ScriptedLLM(), creative or ScriptedLLM(text=OFFER)
    deps = WarRoomDeps(predictor=predictor, retriever=retriever,
                       analytical_llm=lambda: llm, creative_llm=lambda: creative,
                       tools=tools.factory())
    final = asyncio.run(build_graph(deps).ainvoke(initial_state()))
    return final, tools, retriever


def test_low_risk_customer_skips_research_and_mitigation():
    final, tools, retriever = _run(FakePredictor(0.12))
    assert final["risk_label"] == "LOW"
    assert retriever.calls == []                 # agent2 never ran
    assert tools.calls == []                     # agent3 never ran
    assert final["retention_offer"] == ""
    assert [p.split(":")[0] for p in final["agent_path"]] == ["agent1", "FINISH"]


def test_high_risk_runs_all_agents_and_checks_policy_before_offer():
    creative = ScriptedLLM(text=OFFER, script=[
        tool_call("retention_policy_check", {"contract_type": "Month-to-month", "monthly_charge": 80.0}),
    ])
    final, tools, retriever = _run(FakePredictor(0.91), creative=creative)

    assert final["risk_label"] == "HIGH"
    assert len(retriever.calls) == 1
    assert tools.names.index("retention_policy_check") < tools.names.index("log_retention_action")
    assert final["policy"]["max_discount_pct"] == 15
    assert final["retention_offer"].startswith("HOOK:")
    assert final["crm_log_id"] == "RET-TEST-1"
    assert final["agent_path"][-1].startswith("FINISH")


def test_policy_backstop_runs_when_llm_skips_the_tool():
    final, tools, _ = _run(FakePredictor(0.91))   # LLM drafts with no tool calls
    assert "retention_policy_check" in tools.names
    assert final["policy"]["max_discount_pct"] == 15


def test_supervisor_llm_can_request_more_evidence():
    final, _, retriever = _run(FakePredictor(0.91), llm=ScriptedLLM(route="agent2"))
    # The LLM asked to regather; the regather cap (2) then forces the draft.
    assert len(retriever.calls) == 2
    assert final["retention_offer"]


def test_agent3_hides_log_tool_and_answers_repeat_calls_from_cache():
    args = {"contract_type": "Month-to-month", "monthly_charge": 80.0}
    creative = ScriptedLLM(text=OFFER, script=[
        tool_call("retention_policy_check", args, "c1"),
        tool_call("retention_policy_check", args, "c2"),     # identical repeat
    ])
    final, tools, _ = _run(FakePredictor(0.91), creative=creative)
    offered = [s["function"]["name"] for s in creative.bound_specs]
    assert "log_retention_action" not in offered                 # backstop logs instead
    assert tools.names.count("retention_policy_check") == 1      # repeat served from cache
    assert tools.names[-1] == "log_retention_action"
    assert final["crm_log_id"] == "RET-TEST-1"


def test_tool_results_are_compacted_and_capped():
    from churn_agents.agent3_mitigator import _MAX_RESULT_CHARS, _compact_result, _compact_spec
    assert _compact_result('{\n  "a": 1,\n  "b": [1, 2]\n}') == '{"a":1,"b":[1,2]}'
    assert len(_compact_result('"' + "x" * 5000 + '"')) <= _MAX_RESULT_CHARS + 20
    spec = {"type": "function", "function": {"name": "t", "description": "Does X. Then a long story.\n\nArgs: ..."}}
    assert _compact_spec(spec)["function"]["description"] == "Does X"
