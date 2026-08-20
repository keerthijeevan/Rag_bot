"""
Coordinate dense + sparse + hybrid retrieval, metadata filtering, and
reranking into a single call.
"""

from __future__ import annotations

import time

from config.settings import Settings, get_settings
from rag.retrieval.candidate import Candidate
from rag.retrieval.filters import RetrievalFilters, build_qdrant_filter
from rag.retrieval.hybrid_search import hybrid_search
from rag.retrieval.reranker import Reranker


_reranker_cache: dict[str, Reranker] = {}


def _get_reranker(model_name: str) -> Reranker:
    if model_name not in _reranker_cache:
        _reranker_cache[model_name] = Reranker(model_name)
    return _reranker_cache[model_name]


def advanced_search(
    query: str,
    filters: RetrievalFilters | None = None,
    settings: Settings | None = None,
) -> list[Candidate]:
    """Run hybrid retrieval, then rerank candidates."""

    settings = settings or get_settings()

    qdrant_filter = build_qdrant_filter(filters)

    start = time.perf_counter()

    candidates = hybrid_search(
        query,
        limit=settings.top_k,
        query_filter=qdrant_filter,
        settings=settings,
    )

    hybrid_time = time.perf_counter() - start
    print(f"[TIMING] Hybrid retrieval: {hybrid_time:.3f}s")

    if not candidates:
        return []

    reranker = _get_reranker(settings.reranker_model)

    return reranker.rerank(
        query,
        candidates,
        top_k=settings.rerank_top_k,
    )
