"""
Shared pytest fixtures.

Tests that need a live Qdrant instance or a live Gemini API key are marked
with `@pytest.mark.requires_qdrant` / `@pytest.mark.requires_gemini` and are
auto-skipped when those services/credentials aren't configured, so the bulk
of the suite (pure logic: cleaning, chunking, filters, fusion, prompts,
routing, schema validation) always runs without any external dependency.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Settings, get_settings  # noqa: E402


def _qdrant_available() -> bool:
    import socket
    from urllib.parse import urlparse

    settings = get_settings()
    parsed = urlparse(settings.qdrant_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _gemini_available() -> bool:
    return bool(
    get_settings().gemini_classification_api_key
    and get_settings().gemini_generation_api_key
    and get_settings().gemini_validation_api_key
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "requires_qdrant: needs a reachable Qdrant instance")
    config.addinivalue_line("markers", "requires_gemini: needs a configured GEMINI_API_KEY")
    config.addinivalue_line("markers", "requires_network: needs outbound network access")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_qdrant = pytest.mark.skip(reason="Qdrant is not reachable at QDRANT_URL")
    skip_gemini = pytest.mark.skip(reason="GEMINI_API_KEY is not configured")

    qdrant_ok = _qdrant_available()
    gemini_ok = _gemini_available()

    for item in items:
        if "requires_qdrant" in item.keywords and not qdrant_ok:
            item.add_marker(skip_qdrant)
        if "requires_gemini" in item.keywords and not gemini_ok:
            item.add_marker(skip_gemini)


@pytest.fixture
def settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()
