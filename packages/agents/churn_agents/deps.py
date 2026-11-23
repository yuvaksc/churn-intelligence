"""churn_agents.deps — everything the war room needs from the outside world.

The graph depends on interfaces, never on concrete clients:

    predictor        churn_ml.predictor.Predictor   (calibrated risk score)
    retriever        churn_rag.retrieve.Retriever  (similar historical churners)
    analytical_llm / creative_llm   () -> chat model (Groq in production)
    tools            () -> async context of ToolGateway (MCP tools for Agent 3)

`default_deps()` wires the production implementations lazily (nothing heavy loads
until a node first runs). Tests pass fakes to `build_graph(deps)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from churn_ml.predictor import Predictor
from churn_rag.retrieve import Retriever


@dataclass
class WarRoomDeps:
    predictor: Predictor | Any       # Any: the lazy proxy in default_deps
    retriever: Retriever | Any
    analytical_llm: Callable[[], Any]
    creative_llm: Callable[[], Any]
    tools: Callable[[], Any]


class _Lazy:
    """Defers building a dependency until first attribute access."""

    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self._obj = None

    def __getattr__(self, name):
        if self._obj is None:
            self._obj = self._factory()
        return getattr(self._obj, name)


def _predictor():
    from churn_ml.predictor import get_predictor
    return get_predictor()


def _retriever():
    from churn_rag.retrieve import PgHybridRetriever
    return PgHybridRetriever()


# Heavy objects are built once per process and shared by every run.
_SHARED_PREDICTOR = _Lazy(_predictor)
_SHARED_RETRIEVER = _Lazy(_retriever)


def default_deps() -> WarRoomDeps:
    from churn_agents.llm_config import get_analytical_llm, get_creative_llm
    from churn_agents.tools.gateway import mcp_gateway
    return WarRoomDeps(
        predictor=_SHARED_PREDICTOR,
        retriever=_SHARED_RETRIEVER,
        analytical_llm=get_analytical_llm,
        creative_llm=get_creative_llm,
        tools=mcp_gateway,
    )
