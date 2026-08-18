"""
Shared retrieval candidate type.

Every retrieval stage (dense, sparse, hybrid, filtered, reranked, web
fallback) operates on and returns lists of `Candidate`. Kept as its own
tiny module (rather than defined inside dense_search.py or retriever.py)
because multiple sibling modules need it and none of them should import
from each other just to get a type definition.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Candidate:
    """A single piece of retrieved evidence, from Qdrant or web fallback."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    source_type: str  # pdf | docx | txt | markdown | csv | web
    title: str | None = None
    url: str | None = None
    page: int | None = None
    source_ref: str | None = None
    retrieval_method: str = "dense"  # dense | sparse | hybrid | web_fallback
    rerank_score: float | None = None

    def to_source_dict(self) -> dict:
        """Shape returned to the client in the API response's `sources` list."""
        source = {
            "title": self.title or self.source_ref or self.document_id,
            "source_type": self.source_type,
        }
        if self.url:
            source["url"] = self.url
        if self.page is not None:
            source["page"] = self.page
        return source
