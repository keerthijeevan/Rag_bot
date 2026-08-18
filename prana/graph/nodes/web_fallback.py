"""
LangGraph node: one-time website web fallback.

This node is called only when Qdrant retrieval is insufficient.

The fallback is allowed at most once per request. It records both:

    used_web_fallback
        -> whether the fallback was attempted

    web_fallback_succeeded
        -> whether useful website evidence was found
"""

from __future__ import annotations

from graph.state import ChatState
from rag.retrieval.web_fallback import web_fallback_search


def web_fallback_node(state: ChatState) -> dict:
    # Safety guard: never run the website fallback more than once.
    if state.get("used_web_fallback", False):
        return {
            "used_web_fallback": True,
            "web_fallback_succeeded": False,
            "final_context": [],
        }

    web_candidates = web_fallback_search(state["message"])

    # Web fallback is being used because Qdrant evidence was insufficient.
    # Therefore, do not blindly carry insufficient Qdrant evidence forward.
    succeeded = bool(web_candidates)

    return {
        "final_context": web_candidates,
        "used_web_fallback": True,
        "web_fallback_succeeded": succeeded,
    }
