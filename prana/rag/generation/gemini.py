"""
Gemini integration.

Three separate Gemini API keys are used for three responsibilities:

1. Classification
   -> GEMINI_CLASSIFICATION_API_KEY

2. Generation
   -> GEMINI_GENERATION_API_KEY

3. Validation
   -> GEMINI_VALIDATION_API_KEY
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class GeminiConfigError(RuntimeError):
    """Raised when a required Gemini API key is not configured."""


@lru_cache(maxsize=3)
def _get_client(api_key: str):
    from google import genai

    return genai.Client(api_key=api_key)


def _classification_client(settings: Settings):
    if not settings.gemini_classification_api_key:
        raise GeminiConfigError(
            "GEMINI_CLASSIFICATION_API_KEY is not configured."
        )

    return _get_client(settings.gemini_classification_api_key)


def _generation_client(settings: Settings):
    if not settings.gemini_generation_api_key:
        raise GeminiConfigError(
            "GEMINI_GENERATION_API_KEY is not configured."
        )

    return _get_client(settings.gemini_generation_api_key)


def _validation_client(settings: Settings):
    if not settings.gemini_validation_api_key:
        raise GeminiConfigError(
            "GEMINI_VALIDATION_API_KEY is not configured."
        )

    return _get_client(settings.gemini_validation_api_key)


def _extract_json(text: str) -> dict:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")

        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]

    return json.loads(cleaned.strip())


# ------------------------------------------------------------------
# Classification
# ------------------------------------------------------------------

def classify_message(
    message: str,
    settings: Settings | None = None,
) -> str:
    """
    Return:

        GREETING
        NOT_GREETING

    Classification only determines whether the message is a pure
    greeting/social message.
    """

    from rag.prompts.greeting_prompt import build_classification_prompt

    settings = settings or get_settings()

    client = _classification_client(settings)

    prompt = build_classification_prompt(message)

    response = client.models.generate_content(
        model=settings.gemini_classification_model,
        contents=prompt,
    )

    try:
        parsed = _extract_json(response.text)
        classification = parsed.get("classification", "").upper()

    except (json.JSONDecodeError, AttributeError) as exc:
        logger.warning(
            "Failed to parse classifier output (%s); "
            "defaulting to NOT_GREETING",
            exc,
        )
        classification = "NOT_GREETING"

    if classification not in {"GREETING", "NOT_GREETING"}:
        logger.warning(
            "Unexpected classification '%s'; "
            "defaulting to NOT_GREETING",
            classification,
        )
        classification = "NOT_GREETING"

    return classification


# ------------------------------------------------------------------
# Greeting response
# ------------------------------------------------------------------

def generate_direct_response(
    prompt: str,
    settings: Settings | None = None,
) -> str:
    """Generate a short response for a pure greeting."""

    settings = settings or get_settings()

    client = _generation_client(settings)

    response = client.models.generate_content(
        model=settings.gemini_generation_model,
        contents=prompt,
    )

    return response.text.strip()


# ------------------------------------------------------------------
# RAG generation
# ------------------------------------------------------------------

def generate_answer(
    prompt: str,
    settings: Settings | None = None,
) -> str:
    """Generate the grounded RAG answer from supplied evidence."""

    settings = settings or get_settings()

    client = _generation_client(settings)

    response = client.models.generate_content(
        model=settings.gemini_generation_model,
        contents=prompt,
    )

    return response.text.strip()


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def validate_answer(
    prompt: str,
    settings: Settings | None = None,
) -> dict:
    """Validate whether the generated answer is supported by evidence."""

    settings = settings or get_settings()

    client = _validation_client(settings)

    response = client.models.generate_content(
        model=settings.gemini_validation_model,
        contents=prompt,
    )

    try:
        parsed = _extract_json(response.text)

        return {
            "supported": bool(parsed.get("supported", False)),
            "reason": parsed.get("reason", ""),
        }

    except (json.JSONDecodeError, AttributeError) as exc:
        logger.warning(
            "Failed to parse validation output (%s); "
            "treating as unsupported",
            exc,
        )

        return {
            "supported": False,
            "reason": "validation output could not be parsed",
        }
