from __future__ import annotations

from ingestion.processors.preprocessor import is_usable, preprocess_document, preprocess_documents
from ingestion.schema import RawDocument, content_hash, stable_chunk_id, stable_document_id


def _doc(content: str = "This is a reasonably long piece of website content for testing.") -> RawDocument:
    return RawDocument(
        document_id="doc-1",
        source_type="txt",
        source_ref="/tmp/example.txt",
        title="Example",
        content=content,
        metadata={},
    )


def test_stable_document_id_is_deterministic():
    id1 = stable_document_id("pdf", "/data/file.pdf")
    id2 = stable_document_id("pdf", "/data/file.pdf")
    assert id1 == id2


def test_stable_document_id_differs_by_source_type():
    id1 = stable_document_id("pdf", "/data/file.pdf")
    id2 = stable_document_id("docx", "/data/file.pdf")
    assert id1 != id2


def test_stable_chunk_id_is_deterministic():
    assert stable_chunk_id("doc-1", 0) == stable_chunk_id("doc-1", 0)
    assert stable_chunk_id("doc-1", 0) != stable_chunk_id("doc-1", 1)


def test_content_hash_deterministic_and_sensitive():
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("hello!")


def test_is_usable_rejects_empty_document():
    assert is_usable(_doc(content="")) is False


def test_is_usable_rejects_too_short_document():
    assert is_usable(_doc(content="hi")) is False


def test_is_usable_accepts_normal_document():
    assert is_usable(_doc()) is True


def test_preprocess_document_cleans_and_normalizes_metadata():
    raw = _doc(content="Line one.\n\n\n\nLine two with   extra   spaces.   ")
    result = preprocess_document(raw)
    assert result is not None
    assert "\n\n\n\n" not in result.content
    assert result.metadata["source_type"] == "txt"
    assert result.metadata["title"] == "Example"


def test_preprocess_document_drops_unusable_document():
    raw = _doc(content="   ")
    assert preprocess_document(raw) is None


def test_preprocess_documents_filters_batch():
    docs = [_doc(content="Usable content that is long enough to keep."), _doc(content="")]
    result = preprocess_documents(docs)
    assert len(result) == 1
