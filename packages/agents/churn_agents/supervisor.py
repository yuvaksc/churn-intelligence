"""
churn_agents/supervisor.py — cross-agent supervisor for the War Room graph.

The supervisor is the graph hub: it runs first and regains control after every
specialist. Routing is deterministic where the pipeline is linear (diagnose →
gather → draft → finish) and the LLM is consulted only at the one genuine
decision point — judging whether the gathered evidence is sufficient to draft an
offer, or whether to gather once more. Re-gathering is capped so agent3 always
runs and the loop can never stall.

    [supervisor] ──▶ agent1 / agent2 / agent3 / END
         ▲                  │
         └──────────────────┘   (every specialist returns here)

Deterministic guards:
  - agent1 always runs first (need a risk score before anything else)
  - LOW-risk customers finish immediately (risk-gate / cost guard)
  - a hard MAX_STEPS cap forces FINISH no matter what
"""

from typing import Literal

from pydantic import BaseModel, Field
from langgraph.graph import END

from churn_agents import prompts
from churn_agents.state import WarRoomState
from churn_agents.deps import WarRoomDeps

MAX_STEPS         = 6   # hard cap on routing decisions
MAX_AGENT2_RUNS   = 2   # cap evidence re-gathers before forcing the draft


class _Route(BaseModel):
    """Structured evidence-sufficiency decision from the supervisor LLM."""
    next:   Literal["agent3", "agent2"] = Field(
        description="agent3 to draft the offer now, or agent2 to gather more evidence."
    )
    reason: str = Field(description="One short clause explaining the choice.")


async def _llm_route(state: WarRoomState, allowed: list[str], deps: WarRoomDeps | None) -> str:
    """Ask the LLM to choose among `allowed`; fall back to allowed[0] on any issue."""
    prompt = prompts.messages(prompts.SUFFICIENCY_SYSTEM,
                              prompts.SUFFICIENCY_USER.format(evidence=state.get("evidence_report", "")[:600]))
    try:
        if deps is None:
            from churn_agents.llm_config import get_analytical_llm
            llm = get_analytical_llm()
        else:
            llm = deps.analytical_llm()
        decision = await llm.with_structured_output(_Route).ainvoke(prompt)
        return decision.next if decision.next in allowed else allowed[0]
    except Exception as exc:
        print(f"  [Supervisor] LLM routing failed ({exc}); using fallback")
        return allowed[0]


async def supervisor_node(state: WarRoomState, deps: WarRoomDeps | None = None) -> dict:
    """Decide the next node and append the decision to agent_path."""
    path        = state.get("agent_path", [])
    risk        = state.get("risk_label", "PENDING")
    has_evidence = bool(state.get("evidence_report"))
    has_offer    = bool(state.get("retention_offer"))
    agent2_runs  = sum(1 for p in path if p.startswith("agent2"))

    if risk == "PENDING":
        nxt, why = "agent1", "diagnose first"          # guard: diagnostic first
    elif risk == "LOW":
        nxt, why = "FINISH", "low risk"                # guard: risk-gate / cost guard
    elif len(path) >= MAX_STEPS:
        nxt, why = "FINISH", "max steps"               # guard: hard cap
    elif not has_evidence: 
        nxt, why = "agent2", "gather evidence"         # deterministic: gather first
    elif not has_offer:
        if agent2_runs >= MAX_AGENT2_RUNS:
            nxt, why = "agent3", "draft (regather cap)"     # cap re-gathers → force draft
        else:
            nxt, why = await _llm_route(state, ["agent3", "agent2"], deps), "llm"  # LLM: sufficiency
    else:
        nxt, why = "FINISH", "offer ready"             # offer drafted → done

    print(f"  [Supervisor] -> {nxt}  ({why})")
    return {"next_agent": nxt, "agent_path": path + [f"{nxt}:{why}"]}


def route_from_supervisor(state: WarRoomState) -> str:
    """Conditional-edge function — map the supervisor's decision to a node or END."""
    nxt = state.get("next_agent", "FINISH")
    return END if nxt == "FINISH" else nxt
