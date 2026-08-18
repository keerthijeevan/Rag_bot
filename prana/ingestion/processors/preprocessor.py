"""
Prepare loaded (and cleaned) documents for chunking.

Where cleaner.py handles text-level cleanup, this module handles
document-level structural concerns: validating that a document is usable,
normalizing metadata into a consistent shape, and rejecting documents that
carry no real content.
"""

from __future__ import annotations

import logging

from ingestion.processors.cleaner import clean_text
from ingestion.schema import RawDocument

logger = logging.getLogger(__name__)

MIN_CONTENT_LENGTH = 20  # characters; anything shorter isn't useful knowledge


def normalize_metadata(document: RawDocument) -> dict:
    """Ensure every document carries a consistent, minimal metadata shape
    regardless of source type, so downstream payload construction in
    ingestion/vector_store doesn't need per-source-type branching."""
    metadata = dict(document.metadata)
    metadata.setdefault("source_type", document.source_type)
    metadata.setdefault("title", document.title)
    metadata.setdefault("source_ref", document.source_ref)
    return metadata


def is_usable(document: RawDocument) -> bool:
    """Reject documents that have no meaningful content to index."""
    if not document.content:
        return False
    if len(document.content.strip()) < MIN_CONTENT_LENGTH:
        return False
    return True


def preprocess_document(document: RawDocument) -> RawDocument | None:
    """Clean + validate + normalize a single document.

    Returns None if the document should be dropped (empty/unusable), so
    callers can filter a list with a simple list comprehension.
    """
    cleaned_content = clean_text(document.content)

    candidate = RawDocument(
        document_id=document.document_id,
        source_type=document.source_type,
        source_ref=document.source_ref,
        title=(document.title or "").strip() or document.source_ref,
        content=cleaned_content,
        metadata=document.metadata,
    )

    if not is_usable(candidate):
        logger.info("Dropping unusable document: %s (%s)", candidate.source_ref, candidate.source_type)
        return None

    candidate.metadata = normalize_metadata(candidate)
    return candidate


def preprocess_documents(documents: list[RawDocument]) -> list[RawDocument]:
    """Preprocess a batch, silently dropping unusable documents (logged)."""
    result: list[RawDocument] = []
    for doc in documents:
        processed = preprocess_document(doc)
        if processed is not None:
            result.append(processed)
    return result
