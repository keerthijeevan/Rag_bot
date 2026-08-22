"""
Rerank retrieved candidates through the remote Railway reranker API.

The main RAG backend does not load the CrossEncoder locally.
It sends the retrieved document texts to the remote reranker
and maps the returned indexes and scores back to the original
Candidate objects.
"""

from __future__ import annotations

import time

import httpx

from config.settings import Settings, get_settings
from rag.retrieval.candidate import Candidate


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ettin-reranker-17m-v1",
        settings: Settings | None = None,
    ):
        self.model_name = model_name
        self.settings = settings or get_settings()

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        top_k: int,
    ) -> list[Candidate]:

        if not candidates:
            return []

        # Preserve the existing behavior:
        # only the first 5 candidates are sent for reranking.
        rerank_candidates = candidates[:5]

        documents = [
            candidate.text
            for candidate in rerank_candidates
        ]

        print(
            f"[TIMING] Reranker candidates: "
            f"{len(documents)}"
        )

        payload = {
            "query": query,
            "documents": documents,
            "top_k": top_k,
        }

        start = time.perf_counter()

        response = httpx.post(
            self.settings.reranker_url,
            json=payload,
            timeout=self.settings.reranker_timeout_seconds,
        )

        response.raise_for_status()

        data = response.json()

        rerank_time = time.perf_counter() - start

        print(
            f"[TIMING] Remote reranker request: "
            f"{rerank_time:.3f}s"
        )

        results = data["results"]

        ranked = []

        for result in results:
            index = result["index"]
            score = float(result["score"])

            # IMPORTANT:
            # Use the original Candidate object.
            candidate = rerank_candidates[index]

            candidate.rerank_score = score

            ranked.append(candidate)

        ranked.sort(
            key=lambda c: (
                c.rerank_score
                if c.rerank_score is not None
                else float("-inf")
            ),
            reverse=True,
        )

        return ranked[:top_k]
