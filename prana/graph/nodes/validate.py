"""
LangGraph node: validate that the generated answer is actually supported
by the retrieved evidence.

Similarity between retrieval and generation is not the same as factual
correctness, so this runs a dedicated validation call rather than assuming
a grounded prompt guarantees a grounded answer.
"""

from __future__ import annotations

from graph.state import ChatState
from rag.generation.gemini import validate_answer
from rag.prompts.rag_prompt import build_validation_prompt
from utils.timing import timed

SAFE_INSUFFICIENT_EVIDENCE_RESPONSE = (
    "I don't have enough reliable information in the available knowledge base to answer that "
    "confidently. Could you rephrase the question, or ask something else about the site?"
)

@timed("Validation")
def validate_node(state: ChatState) -> dict:
    context = state.get("final_context", [])
    prompt = build_validation_prompt(state["message"], state["answer"], context)
    result = validate_answer(prompt)

    return {
        "validation_supported": result["supported"],
        "validation_reason": result["reason"],
        "retry_count": state.get("retry_count", 0),
    }


def apply_safe_fallback_node(state: ChatState) -> dict:
    """Reached when validation fails and retries are exhausted -- return a
    safe, honest "insufficient evidence" response rather than an
    unverified answer."""
    return {"answer": SAFE_INSUFFICIENT_EVIDENCE_RESPONSE, "sources": []}


def increment_retry_node(state: ChatState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}
