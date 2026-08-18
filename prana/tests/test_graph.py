from __future__ import annotations

from unittest.mock import patch

from graph.chat_graph import (
    _route_after_classification,
    _route_after_rerank,
    _route_after_validate,
    _route_after_web_fallback,
)
from graph.nodes.direct_response import direct_response_node
from graph.nodes.generate import generate_node
from graph.nodes.validate import (
    apply_safe_fallback_node,
    increment_retry_node,
)
from graph.nodes.web_fallback import web_fallback_node
from rag.retrieval.candidate import Candidate


def test_route_after_classification_greeting_goes_direct():
    assert _route_after_classification(
        {"classification": "GREETING"}
    ) == "direct_response"


def test_route_after_classification_not_greeting_goes_to_retrieve():
    assert _route_after_classification(
        {"classification": "NOT_GREETING"}
    ) == "retrieve"


def test_route_after_rerank_sufficient_evidence_goes_to_generate():
    assert _route_after_rerank(
        {"evidence_sufficient": True}
    ) == "generate"


def test_route_after_rerank_insufficient_evidence_goes_to_web_fallback():
    assert _route_after_rerank(
        {
            "evidence_sufficient": False,
            "used_web_fallback": False,
        }
    ) == "web_fallback"


def test_route_after_rerank_does_not_repeat_web_fallback():
    assert _route_after_rerank(
        {
            "evidence_sufficient": False,
            "used_web_fallback": True,
        }
    ) == "out_of_scope"


def test_route_after_web_fallback_with_evidence_goes_to_generate():
    assert _route_after_web_fallback(
        {"web_fallback_succeeded": True}
    ) == "generate"


def test_route_after_web_fallback_without_evidence_goes_out_of_scope():
    assert _route_after_web_fallback(
        {"web_fallback_succeeded": False}
    ) == "out_of_scope"


def test_route_after_validate_supported_ends():
    assert _route_after_validate(
        {"validation_supported": True}
    ) == "end"


def test_route_after_validate_unsupported_retries_when_budget_remains():
    with patch("graph.chat_graph.get_settings") as mock_settings:
        mock_settings.return_value.max_generation_retries = 2

        assert _route_after_validate(
            {
                "validation_supported": False,
                "retry_count": 0,
            }
        ) == "retry"


def test_route_after_validate_unsupported_falls_back_when_retries_exhausted():
    with patch("graph.chat_graph.get_settings") as mock_settings:
        mock_settings.return_value.max_generation_retries = 1

        assert _route_after_validate(
            {
                "validation_supported": False,
                "retry_count": 1,
            }
        ) == "safe_fallback"


def test_direct_response_node_greeting_calls_generation(monkeypatch):
    monkeypatch.setattr(
        "graph.nodes.direct_response.generate_direct_response",
        lambda prompt: "Hi there!",
    )

    result = direct_response_node(
        {
            "classification": "GREETING",
            "message": "hello",
        }
    )

    assert result["answer"] == "Hi there!"
    assert result["sources"] == []


def test_out_of_scope_response_node_uses_fixed_response():
    from graph.nodes.out_of_scope_response import out_of_scope_response_node

    result = out_of_scope_response_node({"message": "what's the weather"})

    assert "website" in result["answer"].lower()
    assert result["sources"] == []


def test_generate_node_produces_sources_from_context(monkeypatch):
    monkeypatch.setattr(
        "graph.nodes.generate.generate_answer",
        lambda prompt: "The answer.",
    )

    candidate = Candidate(
        chunk_id="c1",
        document_id="d1",
        text="evidence text",
        score=0.9,
        source_type="pdf",
        title="Doc Title",
        page=2,
    )

    result = generate_node(
        {
            "message": "q",
            "final_context": [candidate],
        }
    )

    assert result["answer"] == "The answer."


def test_web_fallback_node_uses_web_evidence_only(monkeypatch):
    fallback_result = [
        Candidate(
            chunk_id="b",
            document_id="d2",
            text="t2",
            score=0.5,
            source_type="web",
        )
    ]

    monkeypatch.setattr(
        "graph.nodes.web_fallback.web_fallback_search",
        lambda message, **kw: fallback_result,
    )

    result = web_fallback_node(
        {
            "message": "q",
            "used_web_fallback": False,
        }
    )

    assert result["used_web_fallback"] is True
    assert result["web_fallback_succeeded"] is True
    assert result["final_context"] == fallback_result


def test_web_fallback_node_marks_failure_when_no_web_evidence(monkeypatch):
    monkeypatch.setattr(
        "graph.nodes.web_fallback.web_fallback_search",
        lambda message, **kw: [],
    )

    result = web_fallback_node(
        {
            "message": "q",
            "used_web_fallback": False,
        }
    )

    assert result["used_web_fallback"] is True
    assert result["web_fallback_succeeded"] is False
    assert result["final_context"] == []


def test_web_fallback_node_does_not_run_twice(monkeypatch):
    called = False

    def fake_search(message, **kw):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(
        "graph.nodes.web_fallback.web_fallback_search",
        fake_search,
    )

    result = web_fallback_node(
        {
            "message": "q",
            "used_web_fallback": True,
        }
    )

    assert called is False
    assert result["used_web_fallback"] is True
    assert result["web_fallback_succeeded"] is False


def test_increment_retry_node_increments_counter():
    result = increment_retry_node({"retry_count": 1})
    assert result["retry_count"] == 2


def test_increment_retry_node_defaults_to_zero_then_one():
    result = increment_retry_node({})
    assert result["retry_count"] == 1


def test_apply_safe_fallback_node_returns_safe_message():
    result = apply_safe_fallback_node({})
    assert "don't have enough" in result["answer"].lower()
