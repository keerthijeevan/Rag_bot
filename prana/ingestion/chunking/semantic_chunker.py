"""
Optional semantic chunking strategy.

Splits text at points where consecutive sentences diverge semantically
(using dense sentence embeddings), rather than at a fixed character count.
This can produce more coherent chunks for prose-heavy content, but is more
expensive and not automatically better -- it's selected via
`settings.use_semantic_chunking`, defaulting to off.

Uses langchain_experimental's SemanticChunker (an established
implementation) rather than a hand-rolled semantic splitter.
"""

from __future__ import annotations

import logging

from ingestion.schema import Chunk, RawDocument, stable_chunk_id

logger = logging.getLogger(__name__)


def chunk_document_semantic(
    document: RawDocument,
    embedding_model_name: str = "BAAI/bge-small-en-v1.5",
    fallback_chunk_size: int = 800,
    fallback_chunk_overlap: int = 120,
) -> list[Chunk]:
    """Semantically chunk one document. Falls back to recursive chunking
    if the semantic-chunking dependency isn't available or the document is
    too short to benefit from it."""
    text = document.content
    if not text.strip():
        return []

    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        splitter = SemanticChunker(embeddings)
        pieces = splitter.split_text(text)
    except ImportError:
        logger.warning(
            "langchain_experimental/langchain_huggingface not installed; "
            "falling back to recursive chunking for %s",
            document.source_ref,
        )
        from ingestion.chunking.chunker import chunk_document

        return chunk_document(document, chunk_size=fallback_chunk_size, chunk_overlap=fallback_chunk_overlap)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Semantic chunking failed for %s (%s); falling back to recursive", document.source_ref, exc)
        from ingestion.chunking.chunker import chunk_document

        return chunk_document(document, chunk_size=fallback_chunk_size, chunk_overlap=fallback_chunk_overlap)

    chunks: list[Chunk] = []
    for index, piece in enumerate(pieces):
        if not piece.strip():
            continue
        metadata = {
            "document_id": document.document_id,
            "source_type": document.source_type,
            "source_ref": document.source_ref,
            "title": document.title,
            "chunk_index": index,
            "chunking_strategy": "semantic",
        }
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


def chunk_documents_semantic(
    documents: list[RawDocument],
    embedding_model_name: str = "BAAI/bge-small-en-v1.5",
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for document in documents:
        all_chunks.extend(chunk_document_semantic(document, embedding_model_name=embedding_model_name))
    return all_chunks
