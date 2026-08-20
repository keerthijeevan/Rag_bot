"""
LangGraph node: classify the incoming message.

Thin wrapper around rag/generation/gemini.py's classify_message -- this
node's only job is to fit that call into the graph's state shape.
"""

from __future__ import annotations

from graph.state import ChatState
from rag.generation.gemini import classify_message
from utils.timing import timed

@timed("Classification")
def classify_message_node(state: ChatState) -> dict:
    classification = classify_message(state["message"])
    return {"classification": classification}
