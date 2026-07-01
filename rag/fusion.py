"""
rag/fusion.py — Reciprocal Rank Fusion (RRF).

Combine several ranked id-lists (e.g. dense + sparse) into one fused ranking.
RRF score for a document = sum over rankings of 1 / (k + rank), with rank
starting at 1. k dampens the influence of lower ranks (Cormack et al., 2009).
"""


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Return (id, fused_score) pairs sorted by score descending."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    