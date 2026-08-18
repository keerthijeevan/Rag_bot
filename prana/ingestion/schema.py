"""
Shared data structures passed between loaders -> processors -> chunking ->
embeddings -> vector_store.

Kept in ingestion/ (not utils.py / common.py) since these types describe the
shape of ingested knowledge, which is squarely ingestion's responsibility.
Retrieval imports these read-only types when it needs to describe candidates.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any


def stable_document_id(source_type: str, source_ref: str) -> str:
    """Deterministic id for a document, so re-ingesting the same source
    (same file path / URL) overwrites rather than duplicates its vectors."""
    key = f"{source_type}:{source_ref}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def stable_chunk_id(document_id: str, chunk_index: int) -> str:
    key = f"{document_id}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class RawDocument:
    """Output of a loader: one logical document (a file, or one web page)
    before cleaning/chunking."""

    document_id: str
    source_type: str  # "pdf" | "docx" | "txt" | "markdown" | "csv" | "web"
    source_ref: str  # file path or URL
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A single chunk of a document, ready for embedding."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddedChunk:
    """A chunk with its dense vector and sparse representation attached,
    ready to be upserted into Qdrant."""

    chunk: Chunk
    dense_vector: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]
