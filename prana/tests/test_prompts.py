from __future__ import annotations

from rag.prompts.greeting_prompt import build_classification_prompt, build_greeting_prompt, OUT_OF_SCOPE_RESPONSE
from rag.prompts.rag_prompt import build_rag_prompt, build_validation_prompt
from rag.retrieval.candidate import Candidate


def test_classification_prompt_includes_message():
    prompt = build_classification_prompt("Hello there!")
    assert "Hello there!" in prompt
    assert "GREETING" in prompt
    assert "KNOWLEDGE_QUERY" in prompt
    assert "OUT_OF_SCOPE" in prompt


def test_greeting_prompt_includes_message():
    prompt = build_greeting_prompt("hi")
    assert "hi" in prompt


def test_out_of_scope_response_is_nonempty_and_safe():
    assert len(OUT_OF_SCOPE_RESPONSE) > 0
    assert "website" in OUT_OF_SCOPE_RESPONSE.lower()


def test_rag_prompt_includes_question_and_evidence():
    candidates = [
        Candidate(
            chunk_id="c1", document_id="d1", text="Refunds within 30 days.", score=0.9, source_type="web",
            url="https://example.com/refunds",
        )
    ]
    prompt = build_rag_prompt("What is your refund policy?", candidates)
    assert "What is your refund policy?" in prompt
    assert "Refunds within 30 days." in prompt
    assert "https://example.com/refunds" in prompt
    assert "only" in prompt.lower()  # grounding instruction present


def test_rag_prompt_handles_no_evidence():
    prompt = build_rag_prompt("What is your refund policy?", [])
    assert "no evidence retrieved" in prompt


def test_validation_prompt_includes_answer_and_evidence():
    candidates = [Candidate(chunk_id="c1", document_id="d1", text="Refunds within 30 days.", score=0.9, source_type="web")]
    prompt = build_validation_prompt("What is your refund policy?", "You get 30 days.", candidates)
    assert "You get 30 days." in prompt
    assert "Refunds within 30 days." in prompt
    assert "supported" in prompt.lower()
