"""
LangGraph node: finalize the reranked context for generation.

Reranking itself already happens inside rag/retrieval/retriever.py (via
advanced_search -> reranker.py), as part of one retrieval call -- it is
NOT duplicated here. This node exists as an explicit graph step so the
chat graph's edges clearly show "retrieve -> rerank -> evidence check"
rather than hiding reranking inside the retrieve node, and it also
establishes `final_context` as the field the generation node reads from
(which web_fallback.py's node also writes to, on the fallback path).
"""

from __future__ import annotations

from graph.state import ChatState


def rerank_node(state: ChatState) -> dict:
    return {"final_context": state.get("retrieved_candidates", [])}
