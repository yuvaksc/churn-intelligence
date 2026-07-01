import pytest

from rag.bm25 import BM25Index

DOCS = [
    "fiber optic internet month-to-month high charges",
    "dsl internet two year contract low charges",
    "fiber optic internet two year contract",
    "phone service only no internet",
]
IDS = ["d0", "d1", "d2", "d3"]


def test_ranks_keyword_match_first():
    bm = BM25Index(DOCS, IDS)
    res = bm.search("fiber optic month-to-month", top_k=3)
    assert res and res[0][0] == "d0"


def test_allowed_ids_filter():
    bm = BM25Index(DOCS, IDS)
    res = bm.search("fiber optic", top_k=5, allowed_ids={"d2", "d3"})
    assert res, "expected at least one hit"
    assert all(i in {"d2", "d3"} for i, _ in res)
    assert res[0][0] == "d2"


def test_no_match_returns_empty():
    bm = BM25Index(DOCS, IDS)
    assert bm.search("zzz nonexistent token", top_k=5) == []


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        BM25Index(["a", "b"], ["only-one-id"])
