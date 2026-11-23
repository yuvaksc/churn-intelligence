"""churn_rag.retrieve — what Agent 2 calls to find similar past churners.

    query → embed → hybrid_search() SQL → cross-encoder rerank → top k

    Retriever           the contract Agent 2 depends on (tests pass a fake). Every result
                        is a dict with at least doc_id, document, metadata, similarity.
    PgHybridRetriever   the real one, on pgvector + Postgres full-text search.
    rerank()            cross-encoder reranking of the fused candidates.

hybrid_search() fuses three ranked lists with Reciprocal Rank Fusion, after an optional
contract / internet-type pre-filter:
    dense       embedding similarity of the profile text (pgvector HNSW)
    sparse      keyword match (Postgres full-text search)
    structured  nearest neighbours on tenure, monthly charge and payment method — the
                numbers that text retrieval cannot tell apart ("1 month" vs "13 months")
`near` switches the structured signal on. `mode` and `use_reranker` allow comparing
configurations; the defaults are the best measured one.

A cross-encoder scores the (query, document) pair jointly: far more precise than the
bi-encoder cosine of the first stage, but too slow for the whole corpus, so it only
reorders the fused pool. Model ms-marco-MiniLM-L-6-v2: sentence-transformers (torch)
by default; EMBEDDER=fastembed runs the same model on ONNX Runtime (Vercel).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from churn_rag.embed import get_embedder, to_pgvector

os.environ.setdefault("USE_TF", "0")       # see churn_rag.embed
os.environ.setdefault("USE_TORCH", "1")


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


# ── Cross-encoder reranking ──────────────────────────────────────────────────

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_FASTEMBED_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
_reranker = None
_lock = threading.Lock()   # concurrent first loads crash (see churn_rag.embed.get_embedder)


class _FastEmbedReranker:
    def __init__(self):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self._model = TextCrossEncoder(model_name=_FASTEMBED_MODEL)

    def predict(self, pairs):
        query = pairs[0][0]
        return list(self._model.rerank(query, [doc for _, doc in pairs]))


def _get_reranker():
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                if os.getenv("EMBEDDER") == "fastembed":
                    _reranker = _FastEmbedReranker()
                else:
                    from sentence_transformers import CrossEncoder
                    _reranker = CrossEncoder(_MODEL_NAME)
    return _reranker


def warm_up() -> None:
    """Load the embedder and reranker now instead of inside the first analysis."""
    get_embedder().embed(["warm-up"])
    _get_reranker().predict([("warm-up", "warm-up")])


def rerank(
    query: str,
    candidates: list[dict],
    text_key: str,
    top_n: int,
) -> list[dict]:
    """Score each candidate's text against the query, attach a 'rerank_score',
    and return the top_n candidates sorted by it (descending)."""
    if not candidates:
        return []
    model = _get_reranker()
    scores = model.predict([(query, c[text_key]) for c in candidates])
    for c, s in zip(candidates, scores):
        c["rerank_score"] = round(float(s), 4)
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_n]


# ── Hybrid retrieval ─────────────────────────────────────────────────────────

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
            return rerank(query, candidates, text_key="document", top_n=k)
        return candidates[:k]
