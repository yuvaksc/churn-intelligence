"""churn_rag.pg_retriever — Retriever over pgvector + Postgres full-text search.

    query → embed → hybrid_search() SQL → cross-encoder rerank → top k

hybrid_search() fuses three ranked lists with Reciprocal Rank Fusion, after an optional
contract / internet-type pre-filter:
    dense       embedding similarity of the profile text (pgvector HNSW)
    sparse      keyword match (Postgres full-text search)
    structured  nearest neighbours on tenure, monthly charge and payment method — the
                numbers that text retrieval cannot tell apart ("1 month" vs "13 months")
`near` switches the structured signal on. `mode` and `use_reranker` allow comparing
configurations; the defaults are the best measured one.
"""

from __future__ import annotations

from dataclasses import dataclass

from churn_rag.embed import get_embedder, to_pgvector


@dataclass
class PgHybridRetriever:
    pool: int = 30            # candidates per signal and after fusion
    rrf_k: int = 60
    mode: str = "hybrid"      # dense | sparse | hybrid
    use_reranker: bool = True
    ef_search: int = 100      # HNSW recall/speed knob for filtered queries

    def retrieve(self, query: str, k: int, *, churners_only: bool = True,
                 filters: dict | None = None, near: dict | None = None) -> list[dict]:
        from churn_data.db import tx

        filters, near = filters or {}, near or {}
        qvec = to_pgvector(get_embedder().embed([query])[0])
        with tx() as c:
            c.execute(f"set local hnsw.ef_search = {int(self.ef_search)}")
            rows = c.execute(
                "select * from hybrid_search(%s::vector, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [qvec, query, self.pool, churners_only, self.rrf_k,
                 self.mode in ("dense", "hybrid"), self.mode in ("sparse", "hybrid"),
                 filters.get("contract"), filters.get("internet_service"),
                 near.get("tenure_months"), near.get("monthly_charges"), near.get("payment_method")],
            ).fetchall()

        candidates = [{
            "doc_id": r["doc_id"],
            "document": r["content"],
            "metadata": {**(r["metadata"] or {}), "churn_label": int(r["churn_label"]),
                         "churn_reason": r["churn_reason"] or "", "reason_category": r["reason_category"]},
            "similarity": round(float(r["similarity"]), 4),
            "fusion_score": round(float(r["rrf_score"]), 6),
            "dense_rank": r["dense_rank"], "sparse_rank": r["sparse_rank"],
            "structured_rank": r["structured_rank"],
        } for r in rows]

        if self.use_reranker and candidates:
            from churn_rag.rerank import rerank
            return rerank(query, candidates, text_key="document", top_n=k)
        return candidates[:k]
