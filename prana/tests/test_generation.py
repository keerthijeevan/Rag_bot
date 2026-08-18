from __future__ import annotations

from types import SimpleNamespace

import pytest

from config.settings import Settings
from rag.generation import gemini


class _FakeModels:
    def __init__(self, response_text: str):
        self._response_text = response_text
        self.last_call = None

    def generate_content(self, model, contents):
        self.last_call = {
            "model": model,
            "contents": contents,
        }
        return SimpleNamespace(text=self._response_text)


class _FakeClient:
    def __init__(self, response_text: str):
        self.models = _FakeModels(response_text)


def _settings_with_keys() -> Settings:
    return Settings(
        gemini_classification_api_key="fake-classification-key",
        gemini_generation_api_key="fake-generation-key",
        gemini_validation_api_key="fake-validation-key",
    )


# ------------------------------------------------------------------
# Classification tests
# ------------------------------------------------------------------


def test_classify_message_parses_valid_json(monkeypatch):
    monkeypatch.setattr(
        gemini,
        "_classification_client",
        lambda settings: _FakeClient(
            '{"classification": "GREETING"}'
        ),
    )

    result = gemini.classify_message(
        "hi",
        settings=_settings_with_keys(),
    )

    assert result == "GREETING"


def test_classify_message_handles_code_fenced_json(monkeypatch):
    fenced = '```json\n{"classification": "NOT_GREETING"}\n```'

    monkeypatch.setattr(
        gemini,
        "_classification_client",
        lambda settings: _FakeClient(fenced),
    )

    result = gemini.classify_message(
        "what is your policy?",
        settings=_settings_with_keys(),
    )

    assert result == "NOT_GREETING"


def test_classify_message_defaults_to_not_greeting_on_malformed_output(
    monkeypatch,
):
    monkeypatch.setattr(
        gemini,
        "_classification_client",
        lambda settings: _FakeClient("not json at all"),
    )

    result = gemini.classify_message(
        "something",
        settings=_settings_with_keys(),
    )

    assert result == "NOT_GREETING"


def test_classify_message_defaults_to_not_greeting_on_unexpected_category(
    monkeypatch,
):
    monkeypatch.setattr(
        gemini,
        "_classification_client",
        lambda settings: _FakeClient(
            '{"classification": "BANANA"}'
        ),
    )

    result = gemini.classify_message(
        "something",
        settings=_settings_with_keys(),
    )

    assert result == "NOT_GREETING"


# ------------------------------------------------------------------
# Generation tests
# ------------------------------------------------------------------


def test_generate_answer_returns_stripped_text(monkeypatch):
    monkeypatch.setattr(
        gemini,
        "_generation_client",
        lambda settings: _FakeClient(
            "  Here is the answer.  "
        ),
    )

    result = gemini.generate_answer(
        "prompt",
        settings=_settings_with_keys(),
    )

    assert result == "Here is the answer."


# ------------------------------------------------------------------
# Validation tests
# ------------------------------------------------------------------


def test_validate_answer_parses_supported_true(monkeypatch):
    monkeypatch.setattr(
        gemini,
        "_validation_client",
        lambda settings: _FakeClient(
            '{"supported": true, "reason": "matches evidence"}'
        ),
    )

    result = gemini.validate_answer(
        "prompt",
        settings=_settings_with_keys(),
    )

    assert result == {
        "supported": True,
        "reason": "matches evidence",
    }


def test_validate_answer_fails_closed_on_malformed_output(monkeypatch):
    monkeypatch.setattr(
        gemini,
        "_validation_client",
        lambda settings: _FakeClient("garbage"),
    )

    result = gemini.validate_answer(
        "prompt",
        settings=_settings_with_keys(),
    )

    assert result["supported"] is False


# ------------------------------------------------------------------
# Configuration error tests
# ------------------------------------------------------------------


def test_missing_classification_api_key_raises_config_error():
    settings = Settings(
        gemini_classification_api_key="",
        gemini_generation_api_key="fake-generation-key",
        gemini_validation_api_key="fake-validation-key",
    )

    with pytest.raises(gemini.GeminiConfigError):
        gemini.classify_message(
            "hi",
            settings=settings,
        )


def test_missing_generation_api_key_raises_config_error():
    settings = Settings(
        gemini_classification_api_key="fake-classification-key",
        gemini_generation_api_key="",
        gemini_validation_api_key="fake-validation-key",
    )

    with pytest.raises(gemini.GeminiConfigError):
        gemini.generate_answer(
            "prompt",
            settings=settings,
        )


def test_missing_validation_api_key_raises_config_error():
    settings = Settings(
        gemini_classification_api_key="fake-classification-key",
        gemini_generation_api_key="fake-generation-key",
        gemini_validation_api_key="",
    )

    with pytest.raises(gemini.GeminiConfigError):
        gemini.validate_answer(
            "prompt",
            settings=settings,
        )
