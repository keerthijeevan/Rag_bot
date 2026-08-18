"""
Build Qdrant payload filters from structured filter criteria.

This module only *constructs* qdrant_client Filter objects; the payload
indexes those filters rely on for speed are created once, at ingestion
time, in ingestion/vector_store/payload_index.py.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from qdrant_client.http import models as qmodels


@dataclass
class RetrievalFilters:
    """Optional metadata constraints for a retrieval request. Every field
    maps to a payload field that was indexed in payload_index.py -- do not
    add a field here without also indexing it."""

    source_type: str | None = None  # pdf | docx | txt | markdown | csv | web
    document_id: str | None = None
    category: str | None = None
    content_type: str | None = None
    url: str | None = None

    def is_empty(self) -> bool:
        return all(getattr(self, f.name) is None for f in fields(self))


def build_qdrant_filter(filters: RetrievalFilters | None) -> qmodels.Filter | None:
    """Translate RetrievalFilters into a qdrant_client Filter, or None if
    there's nothing to filter on (avoids sending an empty-but-present
    filter object, which is functionally equivalent but noisier)."""
    if filters is None or filters.is_empty():
        return None

    must: list[qmodels.FieldCondition] = []
    if filters.source_type:
        must.append(qmodels.FieldCondition(key="source_type", match=qmodels.MatchValue(value=filters.source_type)))
    if filters.document_id:
        must.append(qmodels.FieldCondition(key="document_id", match=qmodels.MatchValue(value=filters.document_id)))
    if filters.category:
        must.append(qmodels.FieldCondition(key="category", match=qmodels.MatchValue(value=filters.category)))
    if filters.content_type:
        must.append(
            qmodels.FieldCondition(key="content_type", match=qmodels.MatchValue(value=filters.content_type))
        )
    if filters.url:
        must.append(qmodels.FieldCondition(key="url", match=qmodels.MatchValue(value=filters.url)))

    return qmodels.Filter(must=must)
