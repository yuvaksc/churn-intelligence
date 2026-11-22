"""churn_rag.embed — query/document embeddings (384-d, cosine-normalized).

Default: sentence-transformers all-MiniLM-L6-v2 (the model the Chroma index used).
EMBEDDER=fastembed switches to the same model via ONNX Runtime (no torch), which is
what keeps a serverless Python bundle under its size limit.
"""

from __future__ import annotations

import os
import threading

# transformers also imports TensorFlow when it is installed (~10 s; nothing here uses it).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384


class SentenceTransformerEmbedder:
    name = f"st:{MODEL_NAME}"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(MODEL_NAME)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True, batch_size=64).tolist()


class FastEmbedEmbedder:
    name = f"fastembed:{MODEL_NAME}"

    def __init__(self):
        from fastembed import TextEmbedding
        self._model = TextEmbedding(MODEL_NAME)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]


_embedder = None
_lock = threading.Lock()


def get_embedder():
    """Process-wide singleton. The lock matters: two analyses starting together would
    otherwise load the model twice in parallel threads, and concurrent Hugging Face
    loads crash with "Cannot copy out of meta tensor"."""
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                _embedder = (FastEmbedEmbedder() if os.getenv("EMBEDDER") == "fastembed"
                             else SentenceTransformerEmbedder())
    return _embedder


def to_pgvector(vec: list[float]) -> str:
    """pgvector text literal; avoids a client-side pgvector dependency."""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
