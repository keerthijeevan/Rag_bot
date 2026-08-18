from __future__ import annotations

import pytest

pytestmark = pytest.mark.requires_network


def _dense_embedder_available():
    try:
        from ingestion.embeddings.dense_embedder import DenseEmbedder

        DenseEmbedder().model  # triggers model load/download
        return True
    except Exception:
        return False


def _sparse_embedder_available():
    try:
        from ingestion.embeddings.sparse_embedder import SparseEmbedder

        SparseEmbedder().model
        return True
    except Exception:
        return False


@pytest.fixture(scope="module", autouse=True)
def _skip_if_no_models():
    if not (_dense_embedder_available() and _sparse_embedder_available()):
        pytest.skip("Embedding models could not be loaded (likely no network access in this environment)")


def test_dense_embedder_query_and_document_share_dimension():
    from ingestion.embeddings.dense_embedder import DenseEmbedder

    embedder = DenseEmbedder()
    doc_vectors = embedder.embed_documents(["Our refund policy allows 30 day returns."])
    query_vector = embedder.embed_query("What is the refund policy?")
    assert len(doc_vectors[0]) == len(query_vector) == embedder.dimension


def test_dense_embedder_similar_texts_score_higher_than_unrelated():
    from ingestion.embeddings.dense_embedder import DenseEmbedder
    import numpy as np

    embedder = DenseEmbedder()
    query_vector = np.array(embedder.embed_query("What is your refund policy?"))
    relevant = np.array(embedder.embed_documents(["Refunds are accepted within 30 days of purchase."])[0])
    unrelated = np.array(embedder.embed_documents(["Our office is located in downtown Seattle."])[0])

    assert float(np.dot(query_vector, relevant)) > float(np.dot(query_vector, unrelated))


def test_sparse_embedder_returns_indices_and_values():
    from ingestion.embeddings.sparse_embedder import SparseEmbedder

    embedder = SparseEmbedder()
    vectors = embedder.embed_documents(["Product code XJ-4471 is discontinued."])
    assert len(vectors) == 1
    assert len(vectors[0].indices) > 0
    assert len(vectors[0].indices) == len(vectors[0].values)


def test_sparse_embedder_query_matches_document_terms():
    from ingestion.embeddings.sparse_embedder import SparseEmbedder

    embedder = SparseEmbedder()
    query_sparse = embedder.embed_query("XJ-4471")
    assert len(query_sparse.indices) > 0
