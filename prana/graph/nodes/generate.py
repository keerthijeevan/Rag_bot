"""
LangGraph node: generate a grounded answer from the final retrieved
context (Qdrant evidence, optionally supplemented by web fallback evidence).
"""

from __future__ import annotations

from graph.state import ChatState
from rag.generation.gemini import generate_answer
from rag.prompts.rag_prompt import build_rag_prompt
from utils.timing import timed

@timed("Generation")
def generate_node(state: ChatState) -> dict:
    context = state.get("final_context", [])
    prompt = build_rag_prompt(state["message"], context)
    answer = generate_answer(prompt)

    sources = [c.to_source_dict() for c in context]
    return {"answer": answer, "sources": sources}
