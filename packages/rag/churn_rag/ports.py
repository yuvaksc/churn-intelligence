"""churn_rag.ports — the retrieval contract Agent 2 depends on.

PgHybridRetriever implements it on pgvector; tests use a fake. Every result is a dict
with at least `doc_id`, `document`, `metadata` and `similarity`.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        k: int,
        *,
        churners_only: bool = True,
        filters: dict | None = None,      # hard pre-filter: {"contract": ..., "internet_service": ...}
        near: dict | None = None,         # structured signal: {"tenure_months", "monthly_charges", "payment_method"}
    ) -> list[dict]: ...
