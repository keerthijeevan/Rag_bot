"""
Sparse embeddings for hybrid/lexical retrieval.

Uses FastEmbed's maintained sparse-embedding implementation (BM25 by
default, matching Qdrant's own recommended hybrid-search setup) rather than
a hand-rolled TF-IDF/BM25 implementation.
"""

from __future__ import annotations

from config.settings import get_settings


import threading
from dataclasses import dataclass
from functools import lru_cache

_model_lock = threading.Lock()


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name=model_name)


class SparseEmbedder:
    """Thin wrapper around a cached FastEmbed sparse model."""

    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.sparse_embedding_model

    @property
    def model(self):
        with _model_lock:
            return _load_model(self.model_name)

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        if not texts:
            return []
        results = list(self.model.embed(texts))
        return [SparseVector(indices=r.indices.tolist(), values=r.values.tolist()) for r in results]

    def embed_query(self, text: str) -> SparseVector:
        results = list(self.model.query_embed(text))
        result = results[0]
        return SparseVector(indices=result.indices.tolist(), values=result.values.tolist())
