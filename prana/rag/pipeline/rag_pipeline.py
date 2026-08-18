"""
Thin high-level RAG interface.

`graph/chat_graph.py` is the actual workflow controller. This module is a
minimal service-boundary wrapper around it for callers (e.g. the API route,
or a script) that just want "ask a question, get an answer" without
importing graph internals directly. It intentionally does NOT reimplement
routing, retrieval, or generation -- it only calls `run_chat`.
"""

from __future__ import annotations

from graph.chat_graph import run_chat


def answer_question(message: str) -> dict:
    """Run the full chat workflow for a single, independent question.

    Returns {"answer": str, "sources": list[dict]}.
    """
    return run_chat(message)
