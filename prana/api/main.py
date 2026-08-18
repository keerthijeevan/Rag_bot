"""
FastAPI app entrypoint for the RAG backend (Backend 2).

This service handles ONLY RAG functionality. Authentication, authorization,
and all other application functionality live in the separate Express.js
service (Backend 1) and are intentionally not implemented here.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import health, rag
from config.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Website RAG Backend",
        description="RAG-only backend service. Authentication and non-RAG functionality are handled by Express.js.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["POST", "GET"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(rag.router)

    @app.on_event("startup")
    def on_startup() -> None:
        # Ensure the Qdrant collection/payload indexes exist so the first
        # chat request doesn't fail on a missing collection. Failures here
        # are logged, not fatal -- the service can still serve /health, and
        # a clear error will surface on the first real retrieval attempt.
        try:
            from ingestion.vector_store.qdrant_store import initialize_collection

            initialize_collection(settings=settings)
            logger.info("Qdrant collection '%s' verified/initialized", settings.qdrant_collection_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialize Qdrant collection at startup: %s", exc)

    return app


app = create_app()
