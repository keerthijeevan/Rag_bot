"""
Standard recursive-character chunking, using LangChain's
RecursiveCharacterTextSplitter (an established, maintained implementation)
rather than a hand-rolled splitter.
"""

from __future__ import annotations

from ingestion.schema import Chunk, RawDocument, stable_chunk_id


def _resolve_page_for_offset(page_texts: list[str], offset: int) -> int | None:
    """Given the character offset of a chunk within a `\\f`-joined PDF
    document, figure out which 1-indexed page it falls on."""
    if not page_texts:
        return None
    cursor = 0
    for page_number, page_text in enumerate(page_texts, start=1):
        end = cursor + len(page_text)
        if offset <= end:
            return page_number
        cursor = end + 1  # +1 for the '\f' separator
    return len(page_texts)


def chunk_document(
    document: RawDocument,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Split one document into overlapping chunks, preserving metadata
    (including page numbers for PDFs, when available) on each chunk."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    text = document.content
    if not text.strip():
        return []

    pieces = splitter.split_text(text)

    page_texts = document.metadata.get("page_texts")

    chunks: list[Chunk] = []
    cursor = 0
    for index, piece in enumerate(pieces):
        # Find this piece's approximate offset in the original text so we
        # can attribute it to a PDF page when page_texts is available.
        offset = text.find(piece, cursor)
        if offset == -1:
            offset = cursor
        cursor = offset + max(len(piece) - chunk_overlap, 1)

        metadata = {
            "document_id": document.document_id,
            "source_type": document.source_type,
            "source_ref": document.source_ref,
            "title": document.title,
            "chunk_index": index,
        }
        if page_texts:
            page = _resolve_page_for_offset(page_texts, offset)
            if page is not None:
                metadata["page"] = page

        # Carry over source-specific metadata useful for filtering/citation,
        # without re-embedding the (potentially large) page_texts list.
        for key in ("url", "category", "content_type", "row_index", "file_name"):
            if key in document.metadata:
                metadata[key] = document.metadata[key]

        chunks.append(
            Chunk(
                chunk_id=stable_chunk_id(document.document_id, index),
                document_id=document.document_id,
                chunk_index=index,
                text=piece,
                metadata=metadata,
            )
        )

    return chunks


def chunk_documents(
    documents: list[RawDocument],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for document in documents:
        all_chunks.extend(chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return all_chunks
