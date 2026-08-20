"""
POST /rag/chat

Called by the Express.js backend (which owns authentication) -- not
directly by the browser in the intended architecture. This route does not
perform user authentication itself; it optionally verifies that the
request was actually signed by the trusted Express service (see
`_verify_service_signature`), so FastAPI never blindly trusts a raw
user_id or identity claim placed directly on the request by a client.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time

from fastapi import APIRouter, Header, HTTPException, Request

from api.schemas.rag import ChatRequest, ChatResponse
from config.settings import get_settings
from rag.pipeline.rag_pipeline import answer_question

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


def _verify_service_signature(raw_body: bytes, signature: str | None) -> None:
    """Verify an HMAC-SHA256 signature (hex digest) of the raw request body,
    computed by the Express.js service using the shared secret.

    Express.js side (conceptually):
        signature = hmac_sha256_hex(SERVICE_SHARED_SECRET, raw_json_body)
        send header "X-Service-Signature: <signature>"

    Raises 401 if verification is required and fails/missing. No-ops when
    `require_service_signature` is disabled (e.g. local development).
    """
    settings = get_settings()
    if not settings.require_service_signature:
        return

    if not signature:
        raise HTTPException(status_code=401, detail="Missing service signature")

    expected = hmac.new(
        settings.service_shared_secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="Invalid service signature")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: Request,
    payload: ChatRequest,
    x_service_signature: str | None = Header(default=None),
) -> ChatResponse:
    request_start = time.perf_counter()

    raw_body = await request.body()
    print(raw_body)
    _verify_service_signature(raw_body, x_service_signature)

    try:
        rag_start = time.perf_counter()

        result = answer_question(payload.message)
        print(result)

        rag_time = time.perf_counter() - rag_start
        print(f"RAG pipeline time: {rag_time:.3f}s")

    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error while processing chat request")
        raise HTTPException(
            status_code=500,
            detail="Internal error while generating a response",
        ) from exc

    return ChatResponse(answer=result["answer"])
