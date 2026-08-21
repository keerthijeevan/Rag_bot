"""
Rerank retrieved candidates with a cross-encoder.

Retrieval produces a small candidate set. The cross-encoder then scores
query-document pairs jointly to improve relevance ordering.
"""

from __future__ import annotations

import threading
import time
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

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        top_k: int,
    ) -> list[Candidate]:

        if not candidates:
            return []

        # Only rerank the strongest candidates from retrieval.
        # This reduces expensive CrossEncoder inference.
        rerank_candidates = candidates[:5]

        pairs = [
            (query, candidate.text)
            for candidate in rerank_candidates
        ]

        print(f"[TIMING] Reranker candidates: {len(pairs)}")

        start = time.perf_counter()

        scores = self.model.predict(
            pairs,
            batch_size=32,
            show_progress_bar=False,
        )

        rerank_time = time.perf_counter() - start

        print(
            f"[TIMING] Reranker inference: "
            f"{rerank_time:.3f}s"
        )

        for candidate, score in zip(rerank_candidates, scores):
            candidate.rerank_score = float(score)

        ranked = sorted(
            rerank_candidates,
            key=lambda c: (
                c.rerank_score
                if c.rerank_score is not None
                else float("-inf")
            ),
            reverse=True,
        )

        return ranked[:top_k]
