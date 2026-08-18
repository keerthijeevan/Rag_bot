"""
Dense semantic search: embed the query with BGE, search Qdrant's dense
vector index, return candidates.
"""

from __future__ import annotations

from qdrant_client.http import models as qmodels

from config.settings import Settings, get_settings
from ingestion.embeddings.dense_embedder import DenseEmbedder
from ingestion.vector_store.qdrant_store import dense_query
from rag.retrieval.candidate import Candidate

_embedder_cache: dict[str, DenseEmbedder] = {}


def _get_embedder(model_name: str) -> DenseEmbedder:
    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = DenseEmbedder(model_name)
    return _embedder_cache[model_name]


def dense_search(
    query: str,
    limit: int | None = None,
    query_filter: qmodels.Filter | None = None,
    settings: Settings | None = None,
) -> list[Candidate]:
    """Run dense vector search for `query`, returning up to `limit`
    candidates ordered by similarity score (highest first)."""
    settings = settings or get_settings()
    limit = limit or settings.top_k

    embedder = _get_embedder(settings.dense_embedding_model)
    query_vector = embedder.embed_query(query)

    points = dense_query(query_vector, limit=limit, query_filter=query_filter, settings=settings)

    candidates = []
    for point in points:
        payload = point.payload or {}
        candidates.append(
            Candidate(
                chunk_id=payload.get("chunk_id", str(point.id)),
                document_id=payload.get("document_id", ""),
                text=payload.get("text", ""),
                score=float(point.score),
                source_type=payload.get("source_type", "unknown"),
                title=payload.get("title"),
                url=payload.get("url"),
                page=payload.get("page"),
                source_ref=payload.get("source_ref"),
                retrieval_method="dense",
            )
        )
    return candidates
