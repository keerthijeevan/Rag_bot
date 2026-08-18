"""
GET /rith
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/rith")
def health_check() -> dict:
    return {"status": "ok", "service": "rag-fastapi"}
