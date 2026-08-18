"""
Fuse dense + sparse candidate lists into one ranked candidate set.

Uses weighted Reciprocal Rank Fusion (RRF) -- an established, score-scale-
agnostic fusion strategy (dense cosine scores and sparse BM25 scores are
not directly comparable, so fusing by rank rather than raw score avoids
one method dominating just because its score range happens to be larger).
"""

from __future__ import annotations

from qdrant_client.http import models as qmodels

from config.settings import Settings, get_settings
from rag.retrieval.candidate import Candidate
from rag.retrieval.dense_search import dense_search
from rag.retrieval.sparse_search import sparse_search

_RRF_K = 60  # standard RRF smoothing constant


def reciprocal_rank_fusion(
    dense_results: list[Candidate],
    sparse_results: list[Candidate],
    dense_weight: float = 0.6,
) -> list[Candidate]:
    """Combine two ranked candidate lists into one, weighting each list's
    contribution by `dense_weight` (sparse gets `1 - dense_weight`)."""
    sparse_weight = 1.0 - dense_weight
    fused_scores: dict[str, float] = {}
    by_chunk_id: dict[str, Candidate] = {}

    for rank, candidate in enumerate(dense_results):
        fused_scores[candidate.chunk_id] = fused_scores.get(candidate.chunk_id, 0.0) + dense_weight * (
            1.0 / (_RRF_K + rank + 1)
        )
        by_chunk_id.setdefault(candidate.chunk_id, candidate)

    for rank, candidate in enumerate(sparse_results):
        fused_scores[candidate.chunk_id] = fused_scores.get(candidate.chunk_id, 0.0) + sparse_weight * (
            1.0 / (_RRF_K + rank + 1)
        )
        by_chunk_id.setdefault(candidate.chunk_id, candidate)

    fused: list[Candidate] = []
    for chunk_id, score in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True):
        base = by_chunk_id[chunk_id]
        fused.append(
            Candidate(
                chunk_id=base.chunk_id,
                document_id=base.document_id,
                text=base.text,
                score=score,
                source_type=base.source_type,
                title=base.title,
                url=base.url,
                page=base.page,
                source_ref=base.source_ref,
                retrieval_method="hybrid",
            )
        )
    return fused


def hybrid_search(
    query: str,
    limit: int | None = None,
    query_filter: qmodels.Filter | None = None,
    settings: Settings | None = None,
) -> list[Candidate]:
    settings = settings or get_settings()
    limit = limit or settings.top_k

    dense_results = dense_search(query, limit=limit, query_filter=query_filter, settings=settings)
    sparse_results = sparse_search(query, limit=limit, query_filter=query_filter, settings=settings)

    fused = reciprocal_rank_fusion(dense_results, sparse_results, dense_weight=settings.hybrid_dense_weight)
    return fused[:limit]
