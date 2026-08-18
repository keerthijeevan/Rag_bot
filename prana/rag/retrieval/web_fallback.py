"""
Web fallback retrieval.

Runs ONLY when Qdrant evidence is judged insufficient (see
graph/nodes/web_fallback.py for that decision). Reuses the fetch/extract
primitives from ingestion/loaders/web_loader.py rather than re-implementing
HTTP fetching or HTML parsing here -- this module's own job is retrieval
logic: which pages to look at, and which passages on those pages are
actually relevant to the question.

Hard constraint: every fetch stays within `settings.website_domain`. This
is not a general web search tool.
"""

from __future__ import annotations

import logging
import re

import httpx

from config.settings import Settings, get_settings
from ingestion.embeddings.dense_embedder import DenseEmbedder
from ingestion.loaders.web_loader import discover_links, extract_text, fetch_page
from ingestion.schema import content_hash
from rag.retrieval.candidate import Candidate

logger = logging.getLogger(__name__)

_embedder_cache: dict[str, DenseEmbedder] = {}


def _get_embedder(model_name: str) -> DenseEmbedder:
    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = DenseEmbedder(model_name)
    return _embedder_cache[model_name]


def _split_into_passages(text: str, max_chars: int = 600) -> list[str]:
    """Lightweight paragraph-based split so we can score/return a specific
    relevant passage rather than an entire page of text."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    passages: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(buffer) + len(paragraph) + 1 <= max_chars:
            buffer = f"{buffer}\n{paragraph}".strip()
        else:
            if buffer:
                passages.append(buffer)
            buffer = paragraph
    if buffer:
        passages.append(buffer)
    return passages


def _discover_candidate_urls(seed_url: str, allowed_domain: str, timeout: float) -> list[str]:
    """Find a small set of in-domain URLs worth checking, starting from the
    site's homepage. Best-effort: failures just mean fewer candidate URLs,
    not a hard error."""
    urls = {seed_url}
    try:
        response = fetch_page(seed_url, timeout=timeout)
        response.raise_for_status()
        urls.update(discover_links(response.text, seed_url, allowed_domain))
    except httpx.HTTPError as exc:
        logger.warning("Web fallback: could not fetch seed page %s: %s", seed_url, exc)
    return list(urls)


def web_fallback_search(
    query: str,
    settings: Settings | None = None,
) -> list[Candidate]:
    """Attempt to answer `query` using live pages from the configured
    website. Returns an empty list (never raises) if the fallback itself
    fails or finds nothing relevant -- an empty fallback is a valid,
    expected outcome, not an error."""
    settings = settings or get_settings()

    if not settings.web_fallback_enabled:
        return []

    allowed_domain = settings.website_domain
    timeout = settings.web_fallback_timeout_seconds
    max_pages = settings.web_fallback_max_pages

    candidate_urls = _discover_candidate_urls(settings.website_url, allowed_domain, timeout)[:max_pages]
    if not candidate_urls:
        return []

    embedder = _get_embedder(settings.dense_embedding_model)
    query_vector = embedder.embed_query(query)

    import numpy as np

    scored_passages: list[tuple[float, str, str, str]] = []  # (score, passage, url, title)
    seen_hashes: set[str] = set()

    for url in candidate_urls:
        try:
            response = fetch_page(url, timeout=timeout)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.info("Web fallback: skipping %s (%s)", url, exc)
            continue

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            continue

        title, text = extract_text(response.text, url)
        if not text.strip():
            continue

        passages = _split_into_passages(text)
        if not passages:
            continue

        passage_vectors = embedder.embed_documents(passages)
        for passage, vector in zip(passages, passage_vectors):
            h = content_hash(passage)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            score = float(np.dot(query_vector, vector))  # both vectors are normalized -> cosine similarity
            scored_passages.append((score, passage, url, title))

    if not scored_passages:
        return []

    scored_passages.sort(key=lambda item: item[0], reverse=True)
    top = scored_passages[:5]

    return [
        Candidate(
            chunk_id=f"web_fallback:{content_hash(passage)}",
            document_id=url,
            text=passage,
            score=score,
            source_type="web",
            title=title,
            url=url,
            retrieval_method="web_fallback",
        )
        for score, passage, url, title in top
    ]
