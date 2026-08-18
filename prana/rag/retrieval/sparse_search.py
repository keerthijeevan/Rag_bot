"""
Sparse lexical search: embed the query with the configured sparse model
(BM25 via FastEmbed), search Qdrant's sparse vector index. Better than
dense search for exact terminology, product names, IDs, and technical
codes that dense embeddings tend to blur together.
"""

from __future__ import annotations

from qdrant_client.http import models as qmodels

from config.settings import Settings, get_settings
from ingestion.embeddings.sparse_embedder import SparseEmbedder
from ingestion.vector_store.qdrant_store import sparse_query
from rag.retrieval.candidate import Candidate

_embedder_cache: dict[str, SparseEmbedder] = {}


def _get_embedder(model_name: str) -> SparseEmbedder:
    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = SparseEmbedder(model_name)
    return _embedder_cache[model_name]


def sparse_search(
    query: str,
    limit: int | None = None,
    query_filter: qmodels.Filter | None = None,
    settings: Settings | None = None,
) -> list[Candidate]:
    settings = settings or get_settings()
    limit = limit or settings.top_k

    embedder = _get_embedder(settings.sparse_embedding_model)
    sparse_vector = embedder.embed_query(query)

    if not sparse_vector.indices:
        # Query had no recognizable lexical terms (e.g. pure stopwords).
        return []

    points = sparse_query(
        sparse_vector.indices,
        sparse_vector.values,
        limit=limit,
        query_filter=query_filter,
        settings=settings,
    )

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
                retrieval_method="sparse",
            )
        )
    return candidates
