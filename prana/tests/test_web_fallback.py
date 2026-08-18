from __future__ import annotations

import pytest

from ingestion.loaders.web_loader import _same_domain
from rag.retrieval.web_fallback import _split_into_passages, web_fallback_search
from config.settings import Settings


def test_same_domain_matches_exact_domain():
    assert _same_domain("https://example.com/page", "example.com") is True


def test_same_domain_matches_subdomain():
    assert _same_domain("https://blog.example.com/page", "example.com") is True


def test_same_domain_rejects_different_domain():
    assert _same_domain("https://evil.com/page", "example.com") is False


def test_same_domain_rejects_lookalike_domain():
    # "notexample.com" should NOT match "example.com"
    assert _same_domain("https://notexample.com/page", "example.com") is False


def test_split_into_passages_respects_max_chars():
    text = "\n\n".join([f"Paragraph number {i} with some content." for i in range(20)])
    passages = _split_into_passages(text, max_chars=100)
    assert len(passages) > 1
    for p in passages:
        assert len(p) <= 100 + 60  # allow one paragraph's worth of slack


def test_split_into_passages_handles_empty_text():
    assert _split_into_passages("") == []


def test_web_fallback_disabled_returns_empty_list():
    settings = Settings(web_fallback_enabled=False)
    result = web_fallback_search("What is your refund policy?", settings=settings)
    assert result == []


@pytest.mark.requires_network
def test_web_fallback_never_returns_out_of_domain_results():
    settings = Settings(website_url="https://example.com", web_fallback_max_pages=1)
    results = web_fallback_search("test query", settings=settings)
    for candidate in results:
        assert candidate.url is not None
        assert "example.com" in candidate.url
