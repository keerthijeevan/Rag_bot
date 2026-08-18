"""
High-level retrieval interface.

The LangGraph `retrieve` node calls `retrieve()` and nothing else in this
package directly -- it should never need to know that Qdrant, BGE, or a
cross-encoder are involved. This module also owns the evidence-sufficiency
decision, since "is this evidence good enough" is a retrieval-quality
question, not a workflow-orchestration question.

Important: insufficient evidence means "the knowledge base doesn't cover
this," not "the question is irrelevant." The graph is responsible for
routing an insufficient result to web fallback rather than treating it as
out-of-scope.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import Settings, get_settings
from rag.retrieval.advanced_search import advanced_search
from rag.retrieval.candidate import Candidate
from rag.retrieval.filters import RetrievalFilters


@dataclass
class RetrievalResult:
    candidates: list[Candidate]
    is_sufficient: bool


def evidence_is_sufficient(candidates: list[Candidate], settings: Settings) -> bool:
    """A candidate set counts as sufficient evidence when we have at least
    `min_evidence_chunks` candidates and the top one clears
    `min_rerank_score`. Both thresholds are configurable since "good
    enough" is a product/quality tuning decision, not a constant."""
    if len(candidates) < settings.min_evidence_chunks:
        return False
    top_score = candidates[0].rerank_score
    if top_score is None:
        # Reranking wasn't run (e.g. empty candidate list upstream) --
        # treat as insufficient rather than guessing.
        return False
    return top_score >= settings.min_rerank_score


def retrieve(
    question: str,
    filters: RetrievalFilters | None = None,
    settings: Settings | None = None,
) -> RetrievalResult:
    """Retrieve + rerank evidence for `question`, and judge whether it's
    sufficient to answer from."""
    settings = settings or get_settings()
    candidates = advanced_search(question, filters=filters, settings=settings)
    sufficient = evidence_is_sufficient(candidates, settings)
    return RetrievalResult(candidates=candidates, is_sufficient=sufficient)
