"""
churn_rag/rerank.py — cross-encoder reranking (lazy singleton).

A cross-encoder scores the (query, document) pair jointly, which is far more
precise than the bi-encoder cosine used for first-stage retrieval — but too slow
to run over a whole corpus, so it only reorders the fused candidate pool.

Model: ms-marco-MiniLM-L-6-v2.
  default            sentence-transformers CrossEncoder (torch, local dev / training)
  EMBEDDER=fastembed the same model via ONNX Runtime (no torch) — the serverless path,
                     keeps the Python bundle under Vercel's size limit
"""

import os
import threading

os.environ.setdefault("USE_TF", "0")       # see churn_rag.embed
os.environ.setdefault("USE_TORCH", "1")

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
    from churn_rag.embed import get_embedder
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
