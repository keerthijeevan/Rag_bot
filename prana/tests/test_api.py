"""
Tests for the FastAPI layer: request validation, health, and the /rag/chat
route with the underlying chat pipeline mocked out (so this suite never
needs live Qdrant/Gemini credentials).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.schemas.rag import ChatRequest


# ---------------------------------------------------------------------------
# Pydantic schema validation
# ---------------------------------------------------------------------------

def test_chat_request_accepts_normal_message():
    req = ChatRequest(message="What services does the website provide?")
    assert req.message == "What services does the website provide?"


def test_chat_request_rejects_empty_message():
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_chat_request_rejects_whitespace_only_message():
    with pytest.raises(ValidationError):
        ChatRequest(message="    \n\t  ")


def test_chat_request_rejects_over_length_message(settings):
    too_long = "a" * (settings.max_message_length + 1)
    with pytest.raises(ValidationError):
        ChatRequest(message=too_long)


def test_chat_request_does_not_reject_purely_numeric_but_meaningful_question():
    # "1234" alone might reasonably be considered low-signal, but a real
    # question containing digits must remain valid -- digits alone are not
    # grounds for rejection.
    req = ChatRequest(message="What is product 1234?")
    assert req.message == "What is product 1234?"


def test_chat_request_strips_surrounding_whitespace():
    req = ChatRequest(message="  hello  ")
    assert req.message == "hello"


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_endpoint():
    from api.main import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


# ---------------------------------------------------------------------------
# /rag/chat -- pipeline mocked so no live Gemini/Qdrant call happens
# ---------------------------------------------------------------------------

def test_chat_endpoint_happy_path(monkeypatch):
    monkeypatch.setattr(
        "api.routes.rag.answer_question",
        lambda message: {"answer": "The site offers X and Y.", "sources": [{"title": "FAQ", "source_type": "web"}]},
    )
    from api.main import app

    client = TestClient(app)
    response = client.post("/rag/chat", json={"message": "What does the site offer?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The site offers X and Y."
    assert body["sources"] == [{"title": "FAQ", "source_type": "web", "url": None, "page": None}]


def test_chat_endpoint_rejects_malformed_request_missing_message():
    from api.main import app

    client = TestClient(app)
    response = client.post("/rag/chat", json={})
    assert response.status_code == 422


def test_chat_endpoint_rejects_empty_message():
    from api.main import app

    client = TestClient(app)
    response = client.post("/rag/chat", json={"message": "   "})
    assert response.status_code == 422


def test_chat_endpoint_does_not_leak_internal_exception_details(monkeypatch):
    def _boom(message):
        raise RuntimeError("qdrant connection refused at internal-host:6333 with secret=abc123")

    monkeypatch.setattr("api.routes.rag.answer_question", _boom)
    from api.main import app

    client = TestClient(app)
    response = client.post("/rag/chat", json={"message": "hello there"})
    assert response.status_code == 500
    body = response.json()
    assert "secret" not in body["detail"].lower()
    assert "qdrant" not in body["detail"].lower()


def test_chat_endpoint_rejects_bad_signature_when_required(monkeypatch):
    from config.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("REQUIRE_SERVICE_SIGNATURE", "true")
    monkeypatch.setenv("SERVICE_SHARED_SECRET", "test-secret")
    get_settings.cache_clear()

    monkeypatch.setattr(
        "api.routes.rag.answer_question",
        lambda message: {"answer": "ok", "sources": []},
    )

    from api.main import app

    client = TestClient(app)
    response = client.post(
        "/rag/chat",
        json={"message": "hello"},
        headers={"X-Service-Signature": "not-a-real-signature"},
    )
    assert response.status_code == 401

    get_settings.cache_clear()


def test_chat_endpoint_accepts_valid_signature_when_required(monkeypatch):
    import hashlib
    import hmac
    import json

    from config.settings import get_settings

    monkeypatch.setenv("REQUIRE_SERVICE_SIGNATURE", "true")
    monkeypatch.setenv("SERVICE_SHARED_SECRET", "test-secret")
    get_settings.cache_clear()

    monkeypatch.setattr(
        "api.routes.rag.answer_question",
        lambda message: {"answer": "ok", "sources": []},
    )

    from api.main import app

    client = TestClient(app)
    body = {"message": "hello"}
    raw_body = json.dumps(body).encode("utf-8")
    signature = hmac.new(b"test-secret", raw_body, hashlib.sha256).hexdigest()

    response = client.post(
        "/rag/chat",
        content=raw_body,
        headers={"X-Service-Signature": signature, "Content-Type": "application/json"},
    )
    assert response.status_code == 200

    get_settings.cache_clear()
