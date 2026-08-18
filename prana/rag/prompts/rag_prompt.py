"""
Prompt templates for grounded RAG answer generation.
"""

from __future__ import annotations

from rag.retrieval.candidate import Candidate

SYSTEM_INSTRUCTIONS = """You are a helpful assistant that answers questions strictly using the \
provided evidence from a specific website and its documents.

Rules you MUST follow:
- Use ONLY the supplied evidence to answer. Do not use outside/general knowledge.
- Do not invent facts, numbers, names, or policies that are not present in the evidence.
- If the evidence does not contain enough information to answer confidently, say so plainly \
  (e.g. "I don't have that information in the available knowledge base.") instead of guessing.
- Prefer evidence sourced directly from the website's own pages/documents over anything else.
- Clearly distinguish between what the evidence states and any reasonable inference you draw \
  from it -- do not present an inference as a stated fact.
- Answer directly and concisely. Do not pad the answer with disclaimers beyond what's needed.
- Do not mention evidence numbers, evidence labels, source names, retrieval details, or citations in the final answer.
- Return only the natural-language answer to the user's question.
"""


def _format_evidence(candidates: list[Candidate]) -> str:
    blocks = []
    for i, candidate in enumerate(candidates, start=1):
        origin = candidate.url or candidate.title or candidate.document_id
        page_info = f" (page {candidate.page})" if candidate.page else ""
        blocks.append(f"[Evidence {i} — {origin}{page_info}]\n{candidate.text}")
    return "\n\n".join(blocks)


def build_rag_prompt(question: str, candidates: list[Candidate]) -> str:
    evidence_block = _format_evidence(candidates) if candidates else "(no evidence retrieved)"
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"--- EVIDENCE ---\n{evidence_block}\n--- END EVIDENCE ---\n\n"
        f"Question: {question}\n\n"
        f"Answer the question using only the evidence above."
    )


VALIDATION_INSTRUCTIONS = """You are checking whether a generated answer is actually supported by \
the evidence it was given. Respond with a strict JSON object only, no other text, in this shape:
{"supported": true or false, "reason": "short explanation"}

An answer is "supported" only if every factual claim in it can be traced to the evidence. An \
answer that correctly says the evidence is insufficient also counts as "supported" (it isn't \
making anything up). A similarity between the answer's wording and the evidence's wording is \
NOT sufficient on its own -- check that the actual claims are backed by the evidence, not just \
that similar words appear.
"""


def build_validation_prompt(question: str, answer: str, candidates: list[Candidate]) -> str:
    evidence_block = _format_evidence(candidates) if candidates else "(no evidence retrieved)"
    return (
        f"{VALIDATION_INSTRUCTIONS}\n\n"
        f"Question: {question}\n\n"
        f"--- EVIDENCE ---\n{evidence_block}\n--- END EVIDENCE ---\n\n"
        f"Generated answer: {answer}\n\n"
        f"Is this answer supported by the evidence?"
    )
