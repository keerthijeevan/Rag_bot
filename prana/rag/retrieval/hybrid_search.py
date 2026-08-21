"""
Fuse dense + sparse retrieval into one ranked candidate set.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from heapq import nlargest

from qdrant_client.http import models as qmodels

from config.settings import Settings, get_settings
from rag.retrieval.candidate import Candidate
from rag.retrieval.dense_search import dense_search
from rag.retrieval.sparse_search import sparse_search

_RRF_K = 60


def reciprocal_rank_fusion(
    dense_results: list[Candidate],
    sparse_results: list[Candidate],
    dense_weight: float = 0.6,
    limit: int = 10,
) -> list[Candidate]:

    sparse_weight = 1.0 - dense_weight

    fused_scores: dict[str, float] = {}
    by_chunk_id: dict[str, Candidate] = {}

    for rank, candidate in enumerate(dense_results):
        fused_scores[candidate.chunk_id] = (
            fused_scores.get(candidate.chunk_id, 0.0)
            + dense_weight / (_RRF_K + rank + 1)
        )
        by_chunk_id.setdefault(candidate.chunk_id, candidate)

    for rank, candidate in enumerate(sparse_results):
        fused_scores[candidate.chunk_id] = (
            fused_scores.get(candidate.chunk_id, 0.0)
            + sparse_weight / (_RRF_K + rank + 1)
        )
        by_chunk_id.setdefault(candidate.chunk_id, candidate)

    top_fused = nlargest(
        limit,
        fused_scores.items(),
        key=lambda item: item[1],
    )

    fused: list[Candidate] = []

    for chunk_id, score in top_fused:
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

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=2) as executor:

        dense_future = executor.submit(
            dense_search,
            query,
            limit,
            query_filter,
            settings,
        )

        sparse_future = executor.submit(
            sparse_search,
            query,
            limit,
            query_filter,
            settings,
        )

        dense_results = dense_future.result()
        sparse_results = sparse_future.result()

    retrieval_time = time.perf_counter() - start

    print(
        f"[TIMING] Dense + Sparse parallel retrieval: "
        f"{retrieval_time:.3f}s"
    )

    if not dense_results and not sparse_results:
        return []

    return reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        dense_weight=settings.hybrid_dense_weight,
        limit=limit,
    )
