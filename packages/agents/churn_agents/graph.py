"""
churn_agents/graph.py — Assembles the War Room StateGraph (supervisor-routed).

Topology:
    [supervisor] ──▶ agent1 / agent2 / agent3 / END
         ▲                  │
         └──────────────────┘   (every specialist returns to the supervisor)

The supervisor (churn_agents/supervisor.py) chooses the path at runtime: agent1 always
runs first, LOW-risk customers finish immediately, and HIGH-risk routing is decided
by a Groq LLM (bounded by a max-step cap).

Dependencies (predictor, retriever, LLMs, MCP tools) are injected through
`build_graph(deps)`; the module-level `war_room_graph` uses the production wiring
from `default_deps()`, which loads nothing heavy until a node first runs.
"""

from langgraph.graph import StateGraph, END

from churn_agents.deps import WarRoomDeps, default_deps
from churn_agents.state import WarRoomState
from churn_agents.agent1_diagnostic import agent1_node
from churn_agents.agent2_researcher import agent2_node
from churn_agents.agent3_mitigator import agent3_node
from churn_agents.supervisor import supervisor_node, route_from_supervisor


def _bind(fn, deps: WarRoomDeps):
    """Close a node function over its dependencies (LangGraph nodes take only state)."""
    async def node(state: WarRoomState) -> dict:
        return await fn(state, deps)
    node.__name__ = fn.__name__
    return node


def build_graph(deps: WarRoomDeps | None = None):
    deps = deps or default_deps()
    workflow = StateGraph(WarRoomState)

    workflow.add_node("supervisor", _bind(supervisor_node, deps))
    workflow.add_node("agent1", _bind(agent1_node, deps))
    workflow.add_node("agent2", _bind(agent2_node, deps))
    workflow.add_node("agent3", _bind(agent3_node, deps))

    workflow.set_entry_point("supervisor")

    # Supervisor routes to the chosen specialist (or END)
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"agent1": "agent1", "agent2": "agent2", "agent3": "agent3", END: END},
    )

    # Every specialist hands control back to the supervisor
    workflow.add_edge("agent1", "supervisor")
    workflow.add_edge("agent2", "supervisor")
    workflow.add_edge("agent3", "supervisor")

    return workflow.compile()


# Production graph — imported by the API.
war_room_graph = build_graph()
