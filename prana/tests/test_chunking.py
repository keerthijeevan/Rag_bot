from __future__ import annotations

from ingestion.chunking.chunker import chunk_document, chunk_documents
from ingestion.schema import RawDocument


def _long_document(source_type: str = "txt") -> RawDocument:
    paragraph = "This is a sentence about the website's return policy. " * 20
    content = "\n\n".join([paragraph] * 5)
    return RawDocument(
        document_id="doc-long",
        source_type=source_type,
        source_ref="/tmp/long.txt",
        title="Long Doc",
        content=content,
        metadata={},
    )


def test_chunk_document_respects_configured_size_roughly():
    doc = _long_document()
    chunks = chunk_document(doc, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        # Recursive splitting isn't exact, but should stay in the right ballpark.
        assert len(chunk.text) <= 300 + 100


def test_chunk_document_preserves_metadata():
    doc = _long_document()
    chunks = chunk_document(doc, chunk_size=300, chunk_overlap=50)
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["document_id"] == "doc-long"
        assert chunk.metadata["chunk_index"] == i
        assert chunk.metadata["source_type"] == "txt"
        assert chunk.chunk_id  # non-empty


def test_chunk_document_empty_content_returns_no_chunks():
    doc = RawDocument(document_id="d", source_type="txt", source_ref="x", title="x", content="   ", metadata={})
    assert chunk_document(doc) == []


def test_chunk_documents_batches_multiple_docs():
    docs = [_long_document(), _long_document()]
    docs[1].document_id = "doc-long-2"
    chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=50)
    doc_ids = {c.document_id for c in chunks}
    assert doc_ids == {"doc-long", "doc-long-2"}


def test_chunk_ids_are_unique_within_document():
    doc = _long_document()
    chunks = chunk_document(doc, chunk_size=300, chunk_overlap=50)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
