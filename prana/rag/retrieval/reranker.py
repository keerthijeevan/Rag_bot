"""
Rerank retrieved candidates with a cross-encoder.

Dense/sparse/hybrid retrieval scores query and document independently
(bi-encoder style), which is fast but less accurate. A cross-encoder scores
the (query, document) pair jointly, giving a much stronger relevance signal
-- used here as a second stage over a small candidate set rather than over
the whole collection, since it's far more expensive per-item.
"""

from __future__ import annotations

import threading
from functools import lru_cache

from rag.retrieval.candidate import Candidate

_model_lock = threading.Lock()


@lru_cache(maxsize=2)
def _load_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name

    @property
    def model(self):
        with _model_lock:
            return _load_reranker(self.model_name)

    def rerank(self, query: str, candidates: list[Candidate], top_k: int) -> list[Candidate]:
        """Score every candidate against the query and return the top_k,
        each annotated with `rerank_score`. Candidates are otherwise
        unchanged (score/retrieval_method from the original retrieval
        stage are preserved for debugging/observability)."""
        if not candidates:
            return []

        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)

        ranked = sorted(candidates, key=lambda c: c.rerank_score or float("-inf"), reverse=True)
        return ranked[:top_k]
