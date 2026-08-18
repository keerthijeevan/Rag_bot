"""
Centralized Qdrant communication.

Every other module that needs to talk to Qdrant goes through this file --
no other module should instantiate its own QdrantClient. This keeps
connection setup, payload shape, and point construction in one place.
"""

from __future__ import annotations

import logging
import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config.settings import Settings, get_settings
from ingestion.schema import EmbeddedChunk
from ingestion.vector_store.collection import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, ensure_collection
from ingestion.vector_store.payload_index import ensure_payload_indexes

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _get_cached_client(url: str, api_key: str | None) -> QdrantClient:
    return QdrantClient(
        url=url,
        api_key=api_key or None,
    )


def get_qdrant_client(settings: Settings | None = None) -> QdrantClient:
    """Return a process-wide cached QdrantClient.

    Cached (rather than constructed per-call) so we reuse one connection
    pool for the life of the process.
    """
    s = settings or get_settings()
    return _get_cached_client(s.qdrant_url, s.qdrant_api_key or None)


def build_payload(chunk_metadata: dict, text: str) -> dict:
    """Build the Qdrant point payload from chunk metadata + text.

    This is the single place that decides what payload fields exist, so
    filters.py, reranker.py, and API response formatting can all rely on a
    consistent shape.
    """
    payload = {
        "text": text,
        "document_id": chunk_metadata.get("document_id"),
        "chunk_id": chunk_metadata.get("chunk_id"),
        "chunk_index": chunk_metadata.get("chunk_index"),
        "title": chunk_metadata.get("title"),
        "source_type": chunk_metadata.get("source_type"),
        "source_ref": chunk_metadata.get("source_ref"),
    }
    # Optional fields, only included when present so we don't pollute
    # every point with nulls for source-type-specific metadata.
    for optional_key in ("url", "page", "category", "content_type", "row_index", "file_name"):
        if chunk_metadata.get(optional_key) is not None:
            payload[optional_key] = chunk_metadata[optional_key]
    return payload


def initialize_collection(client: QdrantClient | None = None, settings: Settings | None = None) -> None:
    """Ensure the collection and its payload indexes exist. Call this once
    at ingestion time and at FastAPI startup (cheap no-op if already set up)."""
    settings = settings or get_settings()
    client = client or get_qdrant_client(settings)
    ensure_collection(client, settings.qdrant_collection_name, dense_dim=settings.dense_embedding_dim)
    ensure_payload_indexes(client, settings.qdrant_collection_name)


def upsert_embedded_chunks(
    embedded_chunks: list[EmbeddedChunk],
    client: QdrantClient | None = None,
    settings: Settings | None = None,
) -> int:
    """Upsert a batch of embedded chunks into Qdrant. Returns the number of
    points written. Uses the chunk's stable chunk_id as the point id, so
    re-ingesting the same source overwrites rather than duplicates."""
    if not embedded_chunks:
        return 0

    settings = settings or get_settings()
    client = client or get_qdrant_client(settings)

    points = []
    for item in embedded_chunks:
        chunk = item.chunk
        payload = build_payload({**chunk.metadata, "chunk_id": chunk.chunk_id}, chunk.text)
        points.append(
            qmodels.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector={
                    DENSE_VECTOR_NAME: item.dense_vector,
                    SPARSE_VECTOR_NAME: qmodels.SparseVector(
                        indices=item.sparse_indices,
                        values=item.sparse_values,
                    ),
                },
                payload=payload,
            )
        )

    client.upsert(collection_name=settings.qdrant_collection_name, points=points)
    logger.info("Upserted %d points into '%s'", len(points), settings.qdrant_collection_name)
    return len(points)


def delete_document_vectors(
    document_id: str,
    client: QdrantClient | None = None,
    settings: Settings | None = None,
) -> None:
    """Delete all chunks belonging to a given document_id (e.g. before
    re-ingesting an updated source)."""
    settings = settings or get_settings()
    client = client or get_qdrant_client(settings)
    client.delete(
        collection_name=settings.qdrant_collection_name,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=document_id))]
            )
        ),
    )


def dense_query(
    query_vector: list[float],
    limit: int,
    query_filter: qmodels.Filter | None = None,
    client: QdrantClient | None = None,
    settings: Settings | None = None,
):
    settings = settings or get_settings()
    client = client or get_qdrant_client(settings)
    return client.query_points(
        collection_name=settings.qdrant_collection_name,
        query=query_vector,
        using=DENSE_VECTOR_NAME,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    ).points


def sparse_query(
    sparse_indices: list[int],
    sparse_values: list[float],
    limit: int,
    query_filter: qmodels.Filter | None = None,
    client: QdrantClient | None = None,
    settings: Settings | None = None,
):
    settings = settings or get_settings()
    client = client or get_qdrant_client(settings)
    return client.query_points(
        collection_name=settings.qdrant_collection_name,
        query=qmodels.SparseVector(indices=sparse_indices, values=sparse_values),
        using=SPARSE_VECTOR_NAME,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    ).points
