"""
rag/retriever.py — Query interface for both ChromaDB collections.

Lazy-loads the client + collections on first call.
Call build_index.py once before using this module.

Public API:
    query_similar_profiles(query_text, n_results, churners_only) -> list[dict]
    query_churn_reasons(query_text, n_results)                   -> list[dict]
"""

from pathlib import Path

CHROMA_DIR = Path("data/chroma_db")


import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

_client_instance    = None
_profiles_instance  = None
_reasons_instance   = None

def _ef():
    return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

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
        _profiles_instance = _get_client().get_collection(
            "churn_profiles", embedding_function=_ef()
        )
    return _profiles_instance

def _get_reasons():
    global _reasons_instance
    if _reasons_instance is None:
        _reasons_instance = _get_client().get_collection(
            "churn_reasons", embedding_function=_ef()
        )
    return _reasons_instance

def query_similar_profiles(
    query_text:    str,
    n_results:     int  = 5,
    churners_only: bool = True,
) -> list[dict]:
    """
    Find n_results most similar training customers to the query.

    Args:
        query_text:    Profile summary of the current customer (same format as indexed docs)
        n_results:     Number of results to return
        churners_only: If True, restrict to customers who churned (label=1)

    Returns:
        List of dicts with keys: document, metadata, similarity
    """
    col   = _get_profiles()
    where = {"churn_label": 1} if churners_only else None

    results = col.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "document":   doc,
            "metadata":   meta,
            "similarity": round(1.0 - dist, 4),   # cosine: lower dist = higher similarity
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def query_churn_reasons(
    query_text:     str,
    n_results:      int   = 10,      # fetch more so dedup in agent2 has material
    min_similarity: float = 0.35,    # drop semantically distant reasons
) -> list[dict]:
    """
    Find n_results most semantically relevant churn reasons.

    Returns:
        List of dicts with keys: reason, metadata, similarity
    """
    col = _get_reasons()

    results = col.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    raw = [
        {
            "reason":     doc,
            "metadata":   meta,
            "similarity": round(1.0 - dist, 4),
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
    return [r for r in raw if r["similarity"] >= min_similarity]