"""
Centralized application configuration.

All configuration values are loaded from environment variables / a `.env`
file via pydantic-settings. No module in this project should read
`os.environ` directly or hard-code a secret, URL, or tunable value --
everything flows through `get_settings()` in this file.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Environment / API
    # ------------------------------------------------------------------
    environment: Literal["development", "staging", "production", "test"] = Field(
        default="development",
        description="Deployment environment. Controls debug behavior/logging verbosity.",
    )

    api_host: str = Field(
        default="0.0.0.0",
        description="Host FastAPI binds to.",
    )

    api_port: int = Field(
        default=8000,
        description="Port FastAPI binds to.",
    )

    allowed_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000,http://localhost:5000",
        description="Comma-separated list of allowed CORS origins.",
    )

    # ------------------------------------------------------------------
    # Service-to-service trust
    # ------------------------------------------------------------------
    service_shared_secret: str = Field(
        default="change-me-in-env",
        description="Shared secret used to verify signed requests from the Express.js backend.",
    )

    require_service_signature: bool = Field(
        default=False,
        description="If true, reject requests to /rag/* that lack a valid service signature header.",
    )

    # ------------------------------------------------------------------
    # Website configuration
    # ------------------------------------------------------------------
    website_url: str = Field(
        default="https://example.com",
        description=(
            "Base URL/domain of the website this chatbot is scoped to. "
            "Used for website ingestion, web fallback, and domain restriction."
        ),
    )

    # ------------------------------------------------------------------
    # Gemini
    # ------------------------------------------------------------------
    gemini_classification_api_key: str = Field(
        default="",
        description="Gemini API key used for message classification.",
    )

    gemini_generation_api_key: str = Field(
        default="",
        description="Gemini API key used for answer generation.",
    )

    gemini_validation_api_key: str = Field(
        default="",
        description="Gemini API key used for answer validation.",
    )

    gemini_classification_model: str = Field(
        default="gemini-3.5-flash-lite",
        description="Gemini model used for message classification.",
    )

    gemini_generation_model: str = Field(
        default="gemini-3.1-flash-lite",
        description="Gemini model used for grounded answer generation.",
    )

    gemini_validation_model: str = Field(
        default="gemini-3.1-flash-lite",
        description="Gemini model used for answer validation.",
    )

    # ------------------------------------------------------------------
    # Qdrant
    # ------------------------------------------------------------------
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant instance URL.",
    )

    qdrant_api_key: str = Field(
        default="",
        description="Qdrant API key (Qdrant Cloud).",
    )

    qdrant_collection_name: str = Field(
        default="website_knowledge_base",
        description="Name of the Qdrant collection storing chunk vectors + payload.",
    )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    dense_embedding_model: str = Field(
    default="BAAI/bge-small-en-v1.5",
    description=(
        "Sentence-transformers model used for dense embeddings "
        "(chunks + queries)."
    ),
)

    dense_embedding_dim: int = Field(
        default=384,
        description=(
            "Output dimensionality of dense_embedding_model. "
            "Must match the model."
        ),
    )

    sparse_embedding_model: str = Field(
        default="Qdrant/bm25",
        description=(
            "Sparse embedding model/method (FastEmbed sparse model name) "
            "used for hybrid retrieval."
        ),
    )

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------
    chunk_size: int = Field(
        default=800,
        description="Target chunk size, in characters, for recursive chunking.",
    )

    chunk_overlap: int = Field(
        default=120,
        description="Overlap, in characters, between consecutive chunks.",
    )

    use_semantic_chunking: bool = Field(
        default=False,
        description=(
            "If true, use semantic_chunker.py instead of the recursive "
            "chunker for ingestion."
        ),
    )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    top_k: int = Field(
        default=5,
        description="Number of candidates to retrieve before reranking.",
    )

    rerank_top_k: int = Field(
        default=5,
        description="Number of candidates kept after reranking.",
    )

    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model used to rerank retrieved candidates.",
    )

    hybrid_dense_weight: float = Field(
        default=0.6,
        description=(
            "Relative weight given to dense results during hybrid fusion "
            "(0-1). Sparse gets the remainder."
        ),
    )

    min_rerank_score: float = Field(
        default=0.01,
        description=(
            "Minimum reranker score for the top candidate to be considered "
            "sufficient evidence."
        ),
    )

    min_evidence_chunks: int = Field(
        default=1,
        description=(
            "Minimum number of candidates surviving score/relevance filtering "
            "to consider evidence sufficient."
        ),
    )

    # ------------------------------------------------------------------
    # Web fallback
    # ------------------------------------------------------------------
    web_fallback_enabled: bool = Field(
        default=True,
        description="Master toggle for web fallback retrieval.",
    )

    web_fallback_max_pages: int = Field(
        default=3,
        description=(
            "Maximum number of website pages to fetch during a single "
            "fallback attempt."
        ),
    )

    web_fallback_timeout_seconds: float = Field(
        default=8.0,
        description=(
            "Per-request timeout when fetching pages during web fallback."
        ),
    )

    # ------------------------------------------------------------------
    # Validation / generation
    # ------------------------------------------------------------------
    max_generation_retries: int = Field(
        default=1,
        description=(
            "How many times to retry retrieval+generation if answer "
            "validation fails."
        ),
    )

    max_message_length: int = Field(
        default=2000,
        description="Maximum allowed length of an incoming chat message.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("hybrid_dense_weight")
    @classmethod
    def _validate_weight(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("hybrid_dense_weight must be between 0 and 1")
        return v

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def allowed_origins_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]

    @property
    def website_domain(self) -> str:
        """Bare domain (netloc) derived from website_url."""

        from urllib.parse import urlparse

        parsed = urlparse(self.website_url)

        return parsed.netloc or parsed.path


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""

    return Settings()
