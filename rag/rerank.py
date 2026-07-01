"""
rag/rerank.py — cross-encoder reranking (sentence-transformers, lazy singleton).

A cross-encoder scores the (query, document) pair jointly, which is far more
precise than the bi-encoder cosine used for first-stage retrieval — but too slow
to run over a whole corpus, so it only reorders the fused candidate pool.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (local, CPU, ~80 MB, downloaded to the
HF cache on first use — same pattern as the all-MiniLM embedder).
"""

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(_MODEL_NAME)
    return _reranker


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
