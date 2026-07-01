from rag.fusion import reciprocal_rank_fusion


def test_rrf_combines_rankings():
    fused = reciprocal_rank_fusion([["a", "b", "c", "d"], ["c", "a", "e"]], k=60)
    ids = [i for i, _ in fused]
    assert set(ids[:2]) == {"a", "c"}          # present in both rankings → top
    assert set(ids) == {"a", "b", "c", "d", "e"}


def test_rrf_single_ranking_preserves_order():
    fused = reciprocal_rank_fusion([["x", "y", "z"]], k=60)
    assert [i for i, _ in fused] == ["x", "y", "z"]


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == []
