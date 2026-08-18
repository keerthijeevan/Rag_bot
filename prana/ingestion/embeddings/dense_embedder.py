"""
Dense embeddings via BAAI/bge-small-en-v1.5 (sentence-transformers).

The SAME model is used for both document chunks and queries, since BGE
models are trained so document/query embeddings live in the same space
(BGE recommends a query instruction prefix -- applied only to queries,
never to documents, see `embed_query`).
"""

from __future__ import annotations

from config.settings import get_settings


import threading
from functools import lru_cache

_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model_lock = threading.Lock()


@lru_cache(maxsize=4)
def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class DenseEmbedder:
    """Thin wrapper around a cached SentenceTransformer instance.

    Instantiate one per process (e.g. held by the retriever / ingestion
    pipeline) rather than reloading the model per call.
    """

    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.dense_embedding_model

    @property
    def model(self):
        # lru_cache + a lock keeps model loading thread-safe and ensures
        # the (potentially large) model is only loaded once per process,
        # even if multiple DenseEmbedder instances share the same model_name.
        with _model_lock:
            return _load_model(self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunk texts (no query instruction prefix)."""
        if not texts:
            return []
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single user query, using BGE's recommended query
        instruction prefix so it aligns with how documents were embedded."""
        prefixed = _BGE_QUERY_INSTRUCTION + text
        vector = self.model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return vector.tolist()

    @property
    def dimension(self) -> int:
        return int(self.model.get_embedding_dimension())
