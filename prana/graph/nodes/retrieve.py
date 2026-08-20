"""
LangGraph node: retrieve evidence for a knowledge query.

Delegates entirely to rag/retrieval/retriever.py -- this node does not
know about Qdrant, dense/sparse search, or reranking internals. Note that
retriever.retrieve() already reranks internally, so a separate
graph/nodes/rerank.py node is not needed in the compiled graph (see
graph/chat_graph.py); rerank.py is kept as a standalone node module per
the architecture spec for graphs that want retrieval and reranking as
separate steps.
"""

from __future__ import annotations

from graph.state import ChatState
from rag.retrieval.retriever import retrieve
from utils.timing import timed

@timed("Retrieval + Reranking")
def retrieve_node(state: ChatState) -> dict:
    result = retrieve(state["message"])
    return {
        "retrieved_candidates": result.candidates,
        "evidence_sufficient": result.is_sufficient,
    }
