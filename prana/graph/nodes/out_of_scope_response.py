from __future__ import annotations

from graph.state import ChatState
from rag.prompts.greeting_prompt import OUT_OF_SCOPE_RESPONSE


def out_of_scope_response_node(state: ChatState) -> dict:
    return {
        "answer": OUT_OF_SCOPE_RESPONSE,
        "sources": [],
    }
