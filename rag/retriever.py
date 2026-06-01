"""
rag/retriever.py — In-memory vector search using NumPy.

Replaces ChromaDB with pre-computed embeddings loaded from S3/local.
Uses sentence-transformers to embed queries at search time.
Cosine similarity via NumPy — fast enough for ~6k vectors.

Public API (same as before — agent2 code unchanged):
    query_similar_profiles(query_text, n_results, churners_only) -> list[dict]
    query_churn_reasons(query_text, n_results)                   -> list[dict]
"""

import os
import json
import boto3
import numpy as np
from pathlib import Path

# ── Index loading (lazy, cached) ──────────────────────────────────────────────

_index = None

def _load_index():
    global _index
    if _index is not None:
        return _index

    # In Lambda: download from S3 to /tmp
    # Locally: load from data/vector_index/
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        _load_from_s3()
    else:
        _load_from_local()

    return _index


def _load_from_local():
    global _index
    base = Path("data/vector_index")
    _index = {
        "profiles_embeddings": np.load(base / "profiles_embeddings.npy"),
        "profiles_documents":  json.loads((base / "profiles_documents.json").read_text()),
        "profiles_metadatas":  json.loads((base / "profiles_metadatas.json").read_text()),
        "reasons_embeddings":  np.load(base / "reasons_embeddings.npy"),
        "reasons_documents":   json.loads((base / "reasons_documents.json").read_text()),
        "reasons_metadatas":   json.loads((base / "reasons_metadatas.json").read_text()),
    }
    print(f"[RAG] Loaded index locally — "
          f"{len(_index['profiles_documents'])} profiles, "
          f"{len(_index['reasons_documents'])} reasons")


def _load_from_s3():
    global _index
    import boto3
    s3 = boto3.client("s3")
    bucket = os.environ.get("S3_ARTIFACTS_BUCKET", "churn-prod-artifacts-yuva2004")
    prefix = "index/v1"
    tmp    = Path("/tmp/vector_index")
    tmp.mkdir(exist_ok=True)

    files = [
        "profiles_embeddings.npy",
        "profiles_documents.json",
        "profiles_metadatas.json",
        "reasons_embeddings.npy",
        "reasons_documents.json",
        "reasons_metadatas.json",
    ]
    for f in files:
        local_path = tmp / f
        if not local_path.exists():
            print(f"[RAG] Downloading {f} from S3...")
            s3.download_file(bucket, f"{prefix}/{f}", str(local_path))

    _index = {
        "profiles_embeddings": np.load(tmp / "profiles_embeddings.npy"),
        "profiles_documents":  json.loads((tmp / "profiles_documents.json").read_text()),
        "profiles_metadatas":  json.loads((tmp / "profiles_metadatas.json").read_text()),
        "reasons_embeddings":  np.load(tmp / "reasons_embeddings.npy"),
        "reasons_documents":   json.loads((tmp / "reasons_documents.json").read_text()),
        "reasons_metadatas":   json.loads((tmp / "reasons_metadatas.json").read_text()),
    }
    print(f"[RAG] Loaded index from S3 — "
          f"{len(_index['profiles_documents'])} profiles, "
          f"{len(_index['reasons_documents'])} reasons")


# ── Embedding (sentence-transformers, CPU) ────────────────────────────────────

_st_model = None

def _embed(text: str) -> np.ndarray:
    """Embed a query string using sentence-transformers (local, no API needed)."""
    global _st_model
    if _st_model is None:
        os.environ["HF_HOME"] = "/tmp/hf_cache"
        os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="/tmp/hf_cache")
    return _st_model.encode(text, convert_to_numpy=True).astype(np.float32)

# ── Cosine similarity ─────────────────────────────────────────────────────────

def _cosine_similarity(query_vec: np.ndarray,
                        matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and all rows in matrix."""
    query_norm  = query_vec  / (np.linalg.norm(query_vec)  + 1e-9)
    matrix_norm = matrix     / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    return matrix_norm @ query_norm


# ── Public API ────────────────────────────────────────────────────────────────

def query_similar_profiles(
    query_text:    str,
    n_results:     int  = 5,
    churners_only: bool = True,
) -> list[dict]:
    idx        = _load_index()
    query_vec  = _embed(query_text)
    sims       = _cosine_similarity(query_vec, idx["profiles_embeddings"])
    metadatas  = idx["profiles_metadatas"]
    documents  = idx["profiles_documents"]

    # Filter churners only
    if churners_only:
        mask = np.array([m.get("churn_label", 0) == 1 for m in metadatas])
        sims_filtered = np.where(mask, sims, -1)
    else:
        sims_filtered = sims

    top_idx = np.argsort(sims_filtered)[::-1][:n_results]

    return [
        {
            "document":   documents[i],
            "metadata":   metadatas[i],
            "similarity": round(float(sims[i]), 4),
        }
        for i in top_idx
        if sims_filtered[i] > 0
    ]


def query_churn_reasons(
    query_text:     str,
    n_results:      int   = 10,
    min_similarity: float = 0.35,
) -> list[dict]:
    idx       = _load_index()
    query_vec = _embed(query_text)
    sims      = _cosine_similarity(query_vec, idx["reasons_embeddings"])
    documents = idx["reasons_documents"]
    metadatas = idx["reasons_metadatas"]

    top_idx = np.argsort(sims)[::-1][:n_results]

    return [
        {
            "reason":     documents[i],
            "metadata":   metadatas[i],
            "similarity": round(float(sims[i]), 4),
        }
        for i in top_idx
        if sims[i] >= min_similarity
    ]