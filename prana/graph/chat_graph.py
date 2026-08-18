"""
Chat graph: the LangGraph workflow controller for a single chat request.

Gemini performs only greeting detection.

    GREETING
        -> direct response
        -> END

    NOT_GREETING
        -> retrieve
        -> rerank / relevance check
            |
            +-- sufficient -> generate
            |
            +-- insufficient
                    |
                    +-- web fallback not used yet
                    |       -> web fallback ONCE
                    |       -> useful evidence -> generate
                    |       -> no useful evidence -> out-of-scope response
                    |
                    +-- web fallback already used
                            -> out-of-scope response

After generation:

    generate
        -> validate
            |
            +-- supported -> END
            |
            +-- unsupported + retries left -> retrieve
            |
            +-- retries exhausted -> safe fallback

The web fallback is strictly limited to ONE attempt per request.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, StateGraph

from config.settings import get_settings
from graph.nodes.classify_message import classify_message_node
from graph.nodes.direct_response import direct_response_node
from graph.nodes.out_of_scope_response import out_of_scope_response_node
from graph.nodes.out_of_scope_response import out_of_scope_response_node
from graph.nodes.generate import generate_node
from graph.nodes.rerank import rerank_node
from graph.nodes.retrieve import retrieve_node
from graph.nodes.validate import (
    apply_safe_fallback_node,
    increment_retry_node,
    validate_node,
)
from graph.nodes.web_fallback import web_fallback_node
from graph.state import ChatState


# ------------------------------------------------------------------
# Classification routing
# ------------------------------------------------------------------

def _route_after_classification(state: ChatState) -> str:
    """
    Gemini only returns GREETING or NOT_GREETING.

    GREETING:
        Skip the entire RAG pipeline.

    NOT_GREETING:
        Enter the RAG pipeline.
    """
    if state.get("classification") == "GREETING":
        return "direct_response"

    return "retrieve"


# ------------------------------------------------------------------
# Retrieval / reranking routing
# ------------------------------------------------------------------

def _route_after_rerank(state: ChatState) -> str:
    """
    Decide what to do after retrieval + reranking.

    If Qdrant evidence is sufficient:
        -> generate

    If evidence is insufficient and web fallback has NOT been used:
        -> web fallback

    If evidence is insufficient and web fallback was already used:
        -> out-of-scope response

    This prevents:

        retrieve -> web -> retry -> retrieve -> web -> ...

    """
    if state.get("evidence_sufficient"):
        return "generate"

    if state.get("used_web_fallback", False):
        return "out_of_scope"

    return "web_fallback"


# ------------------------------------------------------------------
# Web fallback routing
# ------------------------------------------------------------------

def _route_after_web_fallback(state: ChatState) -> str:
    """
    The web fallback is allowed only once.

    If useful website evidence was found:
        -> generate

    If nothing useful was found:
        -> fixed out-of-scope response

    No Gemini call is needed for the out-of-scope response.
    """
    if state.get("web_fallback_succeeded", False):
        return "generate"

    return "out_of_scope"


# ------------------------------------------------------------------
# Validation routing
# ------------------------------------------------------------------

def _route_after_validate(state: ChatState) -> str:
    """
    Route after answer validation.

    Supported:
        -> END

    Unsupported with retries remaining:
        -> retrieve again

    Unsupported with retries exhausted:
        -> safe fallback
    """
    if state.get("validation_supported"):
        return "end"

    settings = get_settings()

    if state.get("retry_count", 0) < settings.max_generation_retries:
        return "retry"

    return "safe_fallback"


# ------------------------------------------------------------------
# Build graph
# ------------------------------------------------------------------

def build_chat_graph():
    graph = StateGraph(ChatState)

    # --------------------------------------------------------------
    # Nodes
    # --------------------------------------------------------------

    graph.add_node(
        "classify",
        classify_message_node,
    )

    graph.add_node(
        "direct_response",
        direct_response_node,
    )

    graph.add_node(
        "out_of_scope_response",
        out_of_scope_response_node,
    )

    graph.add_node(
        "retrieve",
        retrieve_node,
    )

    graph.add_node(
        "rerank",
        rerank_node,
    )

    graph.add_node(
        "web_fallback",
        web_fallback_node,
    )

    graph.add_node(
        "generate",
        generate_node,
    )

    graph.add_node(
        "validate",
        validate_node,
    )

    graph.add_node(
        "increment_retry",
        increment_retry_node,
    )

    graph.add_node(
        "safe_fallback",
        apply_safe_fallback_node,
    )

    # --------------------------------------------------------------
    # Entry point
    # --------------------------------------------------------------

    graph.set_entry_point("classify")

    # --------------------------------------------------------------
    # Classification
    #
    # GREETING     -> direct response
    # NOT_GREETING -> RAG
    # --------------------------------------------------------------

    graph.add_conditional_edges(
        "classify",
        _route_after_classification,
        {
            "direct_response": "direct_response",
            "retrieve": "retrieve",
        },
    )

    graph.add_edge(
        "direct_response",
        END,
    )

    # --------------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------------

    graph.add_edge(
        "retrieve",
        "rerank",
    )

    # --------------------------------------------------------------
    # Reranking / relevance decision
    #
    # sufficient
    #     -> generate
    #
    # insufficient + fallback unused
    #     -> web fallback
    #
    # insufficient + fallback already used
    #     -> out-of-scope
    # --------------------------------------------------------------

    graph.add_conditional_edges(
        "rerank",
        _route_after_rerank,
        {
            "generate": "generate",
            "web_fallback": "web_fallback",
            "out_of_scope": "out_of_scope_response",
        },
    )

    # --------------------------------------------------------------
    # One-time website fallback
    # --------------------------------------------------------------

    graph.add_conditional_edges(
        "web_fallback",
        _route_after_web_fallback,
        {
            "generate": "generate",
            "out_of_scope": "out_of_scope_response",
        },
    )

    # --------------------------------------------------------------
    # Generation
    # --------------------------------------------------------------

    graph.add_edge(
        "generate",
        "validate",
    )

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    graph.add_conditional_edges(
        "validate",
        _route_after_validate,
        {
            "end": END,
            "retry": "increment_retry",
            "safe_fallback": "safe_fallback",
        },
    )

    # --------------------------------------------------------------
    # Retry
    #
    # IMPORTANT:
    # used_web_fallback remains True in state.
    #
    # Therefore a retry can NEVER trigger another web fallback.
    # --------------------------------------------------------------

    graph.add_edge(
        "increment_retry",
        "retrieve",
    )

    # --------------------------------------------------------------
    # Safe fallback
    # --------------------------------------------------------------

    graph.add_edge(
        "safe_fallback",
        END,
    )

    return graph.compile()


# ------------------------------------------------------------------
# Compiled graph cache
# ------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_chat_graph():
    """
    Compiled graph is stateless/reusable across requests.

    Compile once per process.
    """
    return build_chat_graph()


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def run_chat(message: str) -> dict:
    """
    Run one independent chat request.

    No conversation history or persistent state is used.
    """
    app = get_chat_graph()

    final_state = app.invoke(
        {
            "message": message,
            "retry_count": 0,
            "used_web_fallback": False,
            "web_fallback_succeeded": False,
        }
    )

    return {
        "answer": final_state.get("answer", ""),
        "sources": final_state.get("sources", []),
    }
