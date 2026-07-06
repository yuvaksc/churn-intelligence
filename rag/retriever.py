"""
rag/retriever.py — Hybrid retrieval over the churn_profiles ChromaDB collection.

Pipeline (per query):
    dense (Chroma vector search) + sparse (in-memory BM25)
        → reciprocal-rank fusion
        → true-cosine scoring of the candidate pool
        → cross-encoder rerank
        → top-N

Public API:
    query_similar_profiles(query_text, n_results, churners_only) -> list[dict]
      each churner profile carries its own documented `churn_reason` in metadata,
      plus rerank_score / bm25_score / fusion_score (downstream ignores the scores).

Lazy-loads the Chroma client/collection, a BM25 index (built once from the stored
documents), and the cross-encoder (see rag/rerank.py).
"""

from pathlib import Path

import numpy as np
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from rag.bm25 import BM25Index
from rag.fusion import reciprocal_rank_fusion
from rag.rerank import rerank

CHROMA_DIR = Path("data/chroma_db")

DENSE_K     = 30    # dense candidates pulled from Chroma
SPARSE_K    = 30    # sparse candidates pulled from BM25
RRF_K       = 60    # reciprocal-rank-fusion constant
RERANK_POOL = 40    # max fused candidates sent to the cross-encoder

_client_instance   = None
_profiles_instance = None
_ef_instance       = None
_profiles_corpus   = None


def _ef():
    global _ef_instance
    if _ef_instance is None:
        _ef_instance = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return _ef_instance


def _get_client():
    global _client_instance
    if _client_instance is None:
        if not CHROMA_DIR.exists():
            raise RuntimeError("ChromaDB not found. Run: python rag/build_index.py")
        _client_instance = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client_instance


def _get_profiles():
    global _profiles_instance
    if _profiles_instance is None:
        _profiles_instance = _get_client().get_collection("churn_profiles", embedding_function=_ef())
    return _profiles_instance


def _load_corpus(collection) -> dict:
    """Pull all docs + metadata once and build a BM25 index over them (cached)."""
    data = collection.get(include=["documents", "metadatas"])
    ids, docs, metas = data["ids"], data["documents"], data["metadatas"]
    return {
        "docs":  dict(zip(ids, docs)),
        "metas": dict(zip(ids, metas)),
        "bm25":  BM25Index(docs, ids),
    }


def _get_profiles_corpus():
    global _profiles_corpus
    if _profiles_corpus is None:
        _profiles_corpus = _load_corpus(_get_profiles())
    return _profiles_corpus


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _hybrid(collection, corpus, query_text, text_key, n_results, where, allowed_ids):
    # 1. Dense ranking via Chroma vector search
    dres = collection.query(
        query_texts=[query_text],
        n_results=DENSE_K,
        where=where,
        include=["distances"],
    )
    dense_rank = list(dres["ids"][0])

    # 2. Sparse ranking via BM25 (same filter applied via allowed_ids)
    sparse_hits  = corpus["bm25"].search(query_text, SPARSE_K, allowed_ids=allowed_ids)
    sparse_rank  = [doc_id for doc_id, _ in sparse_hits]
    sparse_score = dict(sparse_hits)

    # 3. Reciprocal-rank fusion → bounded candidate pool
    fused = reciprocal_rank_fusion([dense_rank, sparse_rank], k=RRF_K)
    pool  = [doc_id for doc_id, _ in fused[:RERANK_POOL]]
    if not pool:
        return []

    # 4. True cosine for every candidate (consistent across dense- and sparse-sourced)
    emb_data = collection.get(ids=pool, include=["embeddings"])
    q_emb    = np.asarray(_ef()([query_text])[0], dtype=np.float32)
    cos = {
        doc_id: round(_cosine(q_emb, np.asarray(emb, dtype=np.float32)), 4)
        for doc_id, emb in zip(emb_data["ids"], emb_data["embeddings"])
    }
    fusion_score = dict(fused)

    candidates = [
        {
            text_key:       corpus["docs"][doc_id],
            "metadata":     corpus["metas"][doc_id],
            "similarity":   cos.get(doc_id, 0.0),
            "bm25_score":   round(float(sparse_score.get(doc_id, 0.0)), 4),
            "fusion_score": round(float(fusion_score.get(doc_id, 0.0)), 6),
        }
        for doc_id in pool
    ]

    # 5. Cross-encoder rerank → top n_results
    return rerank(query_text, candidates, text_key=text_key, top_n=n_results)


def query_similar_profiles(
    query_text:    str,
    n_results:     int  = 5,
    churners_only: bool = True,
) -> list[dict]:
    """
    Hybrid-retrieve the most similar historical customers.

    Returns list[dict] with keys: document, metadata, similarity
    (+ rerank_score, bm25_score, fusion_score).
    """
    corpus      = _get_profiles_corpus()
    where       = {"churn_label": 1} if churners_only else None
    allowed_ids = (
        {doc_id for doc_id, m in corpus["metas"].items() if m.get("churn_label") == 1}
        if churners_only else None
    )
    return _hybrid(_get_profiles(), corpus, query_text, "document", n_results, where, allowed_ids)
