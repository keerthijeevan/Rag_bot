"""
LangGraph workflow state for a single chat request.

This is scratch state for the duration of ONE request only. Nothing here
is persisted after the graph finishes. There is no conversation memory,
conversation_id, history storage, or PostgreSQL state in this workflow.

The state records the information needed to route one request through:

    greeting detection
        -> retrieval
        -> reranking / relevance decision
        -> optional one-time website web fallback
        -> generation
        -> validation

The state is intentionally request-local.
"""

from __future__ import annotations

from typing import TypedDict

from rag.retrieval.candidate import Candidate


class ChatState(TypedDict, total=False):

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    message: str


    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    #
    # Gemini ONLY performs greeting detection.
    #
    # GREETING:
    #     Pure greeting/social message.
    #
    # NOT_GREETING:
    #     Anything substantive that should enter the RAG pipeline.
    #
    # Website scope is NOT decided here.
    # ------------------------------------------------------------------

    classification: str  # GREETING | NOT_GREETING


    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    # Candidates returned from Qdrant after hybrid retrieval + reranking.
    retrieved_candidates: list[Candidate]

    # Whether the Qdrant retrieval produced sufficiently relevant
    # evidence according to the configured reranker threshold.
    evidence_sufficient: bool

    # Whether the live website fallback has already been attempted.
    #
    # This prevents:
    #
    #     retrieve -> web -> retry -> retrieve -> web -> ...
    #
    # The web fallback is allowed at most once per request.
    used_web_fallback: bool

    # Whether the one-time web fallback actually produced useful
    # website evidence.
    #
    # This is separate from used_web_fallback because:
    #
    #     used_web_fallback = True
    #
    # does NOT necessarily mean:
    #
    #     web_fallback_succeeded = True
    #
    web_fallback_succeeded: bool

    # Final evidence passed to the generation stage.
    #
    # This can contain Qdrant evidence, web evidence, or both depending
    # on the retrieval path.
    final_context: list[Candidate]


    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    answer: str


    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    validation_supported: bool
    validation_reason: str

    # Number of generation/validation retry cycles already performed.
    retry_count: int


    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    # Kept internally for compatibility with the current pipeline.
    # The API layer will later decide whether this should be exposed
    # to the frontend.
    sources: list[dict]
