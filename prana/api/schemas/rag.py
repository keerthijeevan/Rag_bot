"""
Pydantic schemas for /rag/chat.

Deliberately minimal: no conversation_id, no history, no previous_messages.
Each request is a single, independent question.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from config.settings import get_settings


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's question or message.")

    @field_validator("message")
    @classmethod
    def _validate_message(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("message must not be empty or whitespace-only")
        max_length = get_settings().max_message_length
        if len(stripped) > max_length:
            raise ValueError(f"message is too long (max {max_length} characters)")
        return stripped


class Source(BaseModel):
    title: str
    source_type: str
    url: str | None = None
    page: int | None = None


class ChatResponse(BaseModel):
    answer: str
    
