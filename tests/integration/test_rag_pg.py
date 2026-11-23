"""pgvector hybrid_search(): RRF fusion, churner filter, metadata pre-filters, the
structured (tenure / charge / payment) signal and the dense / sparse modes. A deterministic bag-of-words embedder stands in for the
real model, so nothing is downloaded."""

import hashlib

import numpy as np
import pytest

import churn_rag.embed as embed
import churn_rag.retrieve as pr
from churn_rag.index import Doc, upsert
from churn_rag.retrieve import PgHybridRetriever


class HashEmbedder:
    name = "test:hash"

    def embed(self, texts):
        out = []
        for t in texts:
            v = np.zeros(embed.DIM)
            for w in t.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % embed.DIM] += 1
            out.append((v / (np.linalg.norm(v) or 1)).tolist())
        return out


DOCS = [
    Doc("d1", "fiber optic month-to-month high charges no tech support", 1, "Competitor offered more data",
        {"contract": "Month-to-month", "internet_service": "Fiber optic", "tenure_months": 40,
         "monthly_charges": 105.0, "payment_method": "Mailed check"}),
    Doc("d2", "fiber optic month-to-month electronic check", 1, "Price too high",
        {"contract": "Month-to-month", "internet_service": "Fiber optic", "tenure_months": 2,
         "monthly_charges": 72.0, "payment_method": "Electronic check"}),
    Doc("d3", "dsl two year contract long tenure", 1, "Moved",
        {"contract": "Two year", "internet_service": "DSL"}),
    Doc("d4", "fiber optic month-to-month high charges", 0, "",
        {"contract": "Month-to-month", "internet_service": "Fiber optic"}),
]
Q = "fiber optic month-to-month high charges"


@pytest.fixture()
def rag(pg, monkeypatch):
    monkeypatch.setattr(pr, "get_embedder", lambda: HashEmbedder())
    upsert(DOCS, HashEmbedder())
    return PgHybridRetriever(use_reranker=False)


def test_hybrid_returns_churners_ranked_by_rrf(rag):
    res = rag.retrieve(Q, 3)
    ids = [r["doc_id"] for r in res]
    assert "d4" not in ids                                   # non-churner filtered out
    assert ids[0] == "d1"                                    # best on both signals
    assert res[0]["dense_rank"] == 1 and res[0]["sparse_rank"] == 1
    assert res[0]["metadata"]["reason_category"] == "competitor"
    scores = [r["fusion_score"] for r in res]
    assert scores == sorted(scores, reverse=True)


def test_non_churners_can_be_included(rag):
    assert "d4" in [r["doc_id"] for r in rag.retrieve(Q, 4, churners_only=False)]


def test_metadata_prefilter_restricts_candidates(rag):
    ids = [r["doc_id"] for r in rag.retrieve("long tenure contract", 5, filters={"contract": "Two year"})]
    assert ids == ["d3"]
    ids = [r["doc_id"] for r in rag.retrieve(Q, 5, filters={"internet_service": "Fiber optic"})]
    assert set(ids) <= {"d1", "d2"}


def test_dense_and_sparse_only_modes(rag):
    dense = PgHybridRetriever(mode="dense", use_reranker=False).retrieve(Q, 3)
    sparse = PgHybridRetriever(mode="sparse", use_reranker=False).retrieve(Q, 3)
    assert dense and all(r["sparse_rank"] is None for r in dense)
    assert sparse and all(r["dense_rank"] is None for r in sparse)


def test_reindex_is_idempotent(rag):
    upsert(DOCS, HashEmbedder())
    from churn_data.db import tx
    with tx() as c:
        assert c.execute("select count(*) as n from rag_documents").fetchone()["n"] == 4


def test_structured_signal_ranks_the_numerically_closest_customer_first(rag):
    """d1 shares more words with the query, but d2 is 1 month and $1 away from a 1-month,
    $71 customer: the structured signal ranks d2 first and lifts its fused score."""
    before = {r["doc_id"]: r for r in rag.retrieve(Q, 2)}
    near = {"tenure_months": 1, "monthly_charges": 71.0, "payment_method": "Electronic check"}
    after = {r["doc_id"]: r for r in rag.retrieve(Q, 2, near=near)}
    assert after["d2"]["structured_rank"] == 1 and after["d1"]["structured_rank"] == 2
    assert before["d2"]["structured_rank"] is None
    assert after["d2"]["fusion_score"] > before["d2"]["fusion_score"]
