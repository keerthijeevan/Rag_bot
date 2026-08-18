from __future__ import annotations

import pytest

from ingestion.vector_store.qdrant_store import build_payload


def test_build_payload_includes_required_fields():
    metadata = {
        "document_id": "doc-1",
        "chunk_id": "chunk-1",
        "chunk_index": 0,
        "title": "Refund Policy",
        "source_type": "web",
        "source_ref": "https://example.com/refunds",
    }
    payload = build_payload(metadata, text="Refunds within 30 days.")
    assert payload["text"] == "Refunds within 30 days."
    assert payload["document_id"] == "doc-1"
    assert payload["source_type"] == "web"


def test_build_payload_omits_absent_optional_fields():
    metadata = {"document_id": "doc-1", "chunk_id": "c1", "source_type": "txt", "title": "T", "source_ref": "r"}
    payload = build_payload(metadata, text="text")
    assert "url" not in payload
    assert "page" not in payload
    assert "category" not in payload


def test_build_payload_includes_present_optional_fields():
    metadata = {
        "document_id": "doc-1",
        "chunk_id": "c1",
        "source_type": "pdf",
        "title": "T",
        "source_ref": "r",
        "page": 4,
        "category": "policies",
    }
    payload = build_payload(metadata, text="text")
    assert payload["page"] == 4
    assert payload["category"] == "policies"


@pytest.mark.requires_qdrant
def test_ensure_collection_is_idempotent():
    from qdrant_client.http import models as qmodels

    from config.settings import get_settings
    from ingestion.vector_store.collection import ensure_collection, verify_collection_config
    from ingestion.vector_store.qdrant_store import get_qdrant_client

    settings = get_settings()
    client = get_qdrant_client(settings)
    ensure_collection(client, "test_collection_idempotent", dense_dim=settings.dense_embedding_dim)
    ensure_collection(client, "test_collection_idempotent", dense_dim=settings.dense_embedding_dim)  # no-op

    assert verify_collection_config(client, "test_collection_idempotent", settings.dense_embedding_dim)
    client.delete_collection("test_collection_idempotent")


@pytest.mark.requires_qdrant
def test_upsert_and_search_round_trip():
    from ingestion.embeddings.dense_embedder import DenseEmbedder
    from ingestion.embeddings.sparse_embedder import SparseEmbedder
    from ingestion.schema import Chunk, EmbeddedChunk
    from ingestion.vector_store.qdrant_store import dense_query, initialize_collection, upsert_embedded_chunks
    from config.settings import get_settings

    settings = get_settings()
    initialize_collection(settings=settings)

    dense = DenseEmbedder(settings.dense_embedding_model)
    sparse = SparseEmbedder(settings.sparse_embedding_model)

    chunk = Chunk(
        chunk_id="test-roundtrip-chunk",
        document_id="test-roundtrip-doc",
        chunk_index=0,
        text="Our test refund policy allows returns within 30 days.",
        metadata={"document_id": "test-roundtrip-doc", "source_type": "txt", "title": "Test", "source_ref": "x"},
    )
    dense_vec = dense.embed_documents([chunk.text])[0]
    sparse_vec = sparse.embed_documents([chunk.text])[0]

    embedded = EmbeddedChunk(
        chunk=chunk, dense_vector=dense_vec, sparse_indices=sparse_vec.indices, sparse_values=sparse_vec.values
    )
    upsert_embedded_chunks([embedded], settings=settings)

    query_vec = dense.embed_query("What is the refund policy?")
    results = dense_query(query_vec, limit=5, settings=settings)
    assert any(r.payload.get("chunk_id") == "test-roundtrip-chunk" for r in results)
