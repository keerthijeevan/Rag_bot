"""
Prompt templates for lightweight, non-RAG interactions.

Gemini is used here ONLY to determine whether the incoming message is a
pure greeting/social message. It must NOT decide whether a question is
in-scope or out-of-scope.

Routing responsibility:

    GREETING
        -> direct response

    NOT_GREETING
        -> enter the RAG pipeline

The RAG retrieval + CrossEncoder layer is responsible for determining
whether a non-greeting question is relevant to the website knowledge.
"""

from __future__ import annotations


# ------------------------------------------------------------------
# Greeting classification
# ------------------------------------------------------------------

CLASSIFICATION_INSTRUCTIONS = """Determine whether the user's message is
a pure greeting or social message.

Classify into exactly one of these categories:

- GREETING:
  A simple hello, hi, good morning, goodbye, thanks, or other short
  social message that contains NO substantive question or information
  request.

- NOT_GREETING:
  Anything that contains a question, information request, explanation
  request, task, instruction, or other substantive request.

Important rules:
- Do NOT decide whether the message is related to the website.
- Do NOT classify anything as OUT_OF_SCOPE.
- Do NOT classify anything as KNOWLEDGE_QUERY.
- Your ONLY job is to determine whether it is a pure greeting/social
  message.
- If the message contains both a greeting and a substantive question,
  classify it as NOT_GREETING.
- When uncertain, classify it as NOT_GREETING so the message can enter
  the RAG pipeline for proper relevance checking.

Examples:

"Hello" -> GREETING
"Good morning" -> GREETING
"Thanks!" -> GREETING
"Bye" -> GREETING
"Hello, how are you?" -> GREETING

"Hello, what are the benefits of exercise?" -> NOT_GREETING
"What is your refund policy?" -> NOT_GREETING
"Tell me about hydration." -> NOT_GREETING
"Write a C++ program." -> NOT_GREETING
"Who is the president?" -> NOT_GREETING

Respond with a strict JSON object only, with no other text:

{"classification": "GREETING" | "NOT_GREETING"}
"""


def build_classification_prompt(message: str) -> str:
    return f"{CLASSIFICATION_INSTRUCTIONS}\n\nUser message: {message}"


# ------------------------------------------------------------------
# Direct greeting response
# ------------------------------------------------------------------

GREETING_RESPONSE_INSTRUCTIONS = """You are a friendly assistant for a
specific website.

The user has sent a pure greeting or social message.

Respond warmly and briefly. Invite the user to ask a question about the
website and its available content.

Do not answer any website-specific question in this response because
there should be no substantive question in a GREETING message.
"""


def build_greeting_prompt(message: str) -> str:
    return f"{GREETING_RESPONSE_INSTRUCTIONS}\n\nUser message: {message}"


# ------------------------------------------------------------------
# Website-only boundary response
# ------------------------------------------------------------------
#
# This is NOT used by the Gemini greeting classifier.
# It is used later by the RAG/retrieval routing layer when the
# CrossEncoder/relevance decision determines that a question is clearly
# outside the website's knowledge domain.
#
# Keeping this response here allows the same fixed boundary message to
# be reused without calling Gemini.
# ------------------------------------------------------------------

OUT_OF_SCOPE_RESPONSE = (
    "I'm specifically designed to answer questions about this website "
    "and its content only."
)
