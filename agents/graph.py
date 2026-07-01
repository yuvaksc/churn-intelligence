"""
agents/graph.py — Assembles the War Room StateGraph (supervisor-routed).

Topology:
    [supervisor] ──▶ agent1 / agent2 / agent3 / END
         ▲                  │
         └──────────────────┘   (every specialist returns to the supervisor)

The supervisor (agents/supervisor.py) chooses the path at runtime: agent1 always
runs first, LOW-risk customers finish immediately, and HIGH-risk routing is decided
by a Groq LLM (bounded by a max-step cap). Compiled once at import time.
"""

from langgraph.graph import StateGraph, END

from agents.state import WarRoomState
from agents.agent1_diagnostic import agent1_node
from agents.agent2_researcher import agent2_node
from agents.agent3_mitigator import agent3_node
from agents.supervisor import supervisor_node, route_from_supervisor


def build_graph():
    workflow = StateGraph(WarRoomState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("agent1", agent1_node)
    workflow.add_node("agent2", agent2_node)
    workflow.add_node("agent3", agent3_node)

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


# Singleton — imported by the API and run_agents.py
war_room_graph = build_graph()
