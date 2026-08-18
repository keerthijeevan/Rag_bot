from __future__ import annotations

import pytest

from config.settings import Settings
from rag.retrieval.candidate import Candidate
from rag.retrieval.retriever import evidence_is_sufficient


def _candidate(chunk_id: str, rerank_score: float | None) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text="text",
        score=0.5,
        source_type="web",
        rerank_score=rerank_score,
    )


def _settings(min_score: float = 0.35, min_chunks: int = 1) -> Settings:
    return Settings(min_rerank_score=min_score, min_evidence_chunks=min_chunks)


def test_evidence_insufficient_when_no_candidates():
    assert evidence_is_sufficient([], _settings()) is False


def test_evidence_insufficient_below_score_threshold():
    candidates = [_candidate("a", rerank_score=0.1)]
    assert evidence_is_sufficient(candidates, _settings(min_score=0.35)) is False


def test_evidence_sufficient_above_score_threshold():
    candidates = [_candidate("a", rerank_score=0.6)]
    assert evidence_is_sufficient(candidates, _settings(min_score=0.35)) is True


def test_evidence_insufficient_when_rerank_score_missing():
    candidates = [_candidate("a", rerank_score=None)]
    assert evidence_is_sufficient(candidates, _settings()) is False


def test_evidence_insufficient_when_below_min_chunk_count():
    candidates = [_candidate("a", rerank_score=0.9)]
    assert evidence_is_sufficient(candidates, _settings(min_chunks=2)) is False


@pytest.mark.requires_network
def test_reranker_orders_by_relevance():
    from rag.retrieval.reranker import Reranker

    try:
        reranker = Reranker()
        reranker.model  # trigger load
    except Exception:
        pytest.skip("Reranker model could not be loaded (likely no network access)")

    candidates = [
        Candidate(chunk_id="a", document_id="d1", text="Our office hours are 9 to 5.", score=0.5, source_type="web"),
        Candidate(
            chunk_id="b",
            document_id="d2",
            text="Refunds are accepted within 30 days of purchase.",
            score=0.5,
            source_type="web",
        ),
    ]
    ranked = reranker.rerank("What is your refund policy?", candidates, top_k=2)
    assert ranked[0].chunk_id == "b"
    assert ranked[0].rerank_score is not None
