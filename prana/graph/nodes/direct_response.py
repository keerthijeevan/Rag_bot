from __future__ import annotations

from graph.state import ChatState
from rag.generation.gemini import generate_direct_response
from rag.prompts.greeting_prompt import build_greeting_prompt


def direct_response_node(state: ChatState) -> dict:
    prompt = build_greeting_prompt(state["message"])
    answer = generate_direct_response(prompt)

    return {
        "answer": answer,
        "sources": [],
    }
