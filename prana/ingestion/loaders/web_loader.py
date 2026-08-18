"""
Fetch pages from the configured website and convert them into RawDocument
objects for ingestion.

This module only knows how to (a) discover in-domain links from a seed page
and (b) fetch + extract clean text from a page. It does not decide *when*
to run (that's ingestion_graph.py) and it is not the web fallback used
during chat (that's rag/retrieval/web_fallback.py, which reuses the fetch
primitives here rather than duplicating them).
"""

from __future__ import annotations

import logging
from urllib.parse import urldefrag, urljoin, urlparse

import httpx

from ingestion.schema import RawDocument, stable_document_id

logger = logging.getLogger(__name__)

_DEFAULT_HEADERS = {
    "User-Agent": "WebsiteRAGBot/1.0 (+https://example.com/bot-info)",
}


def _same_domain(url: str, allowed_domain: str) -> bool:
    netloc = urlparse(url).netloc
    return netloc == allowed_domain or netloc.endswith(f".{allowed_domain}")


def fetch_page(url: str, timeout: float = 10.0) -> httpx.Response:
    """Fetch a single URL. Raises httpx exceptions on network failure;
    callers should catch and handle them (this module never silently
    swallows fetch failures)."""
    with httpx.Client(follow_redirects=True, timeout=timeout, headers=_DEFAULT_HEADERS) as client:
        return client.get(url)


def extract_text(html: str, url: str) -> tuple[str, str]:
    """Extract (title, clean_text) from raw HTML.

    Prefers `trafilatura` (purpose-built content extraction) and falls back
    to a BeautifulSoup-based extraction if trafilatura is unavailable or
    returns nothing useful.
    """
    try:
        import trafilatura

        extracted = trafilatura.extract(html, url=url, include_comments=False, include_tables=True)
        if extracted and extracted.strip():
            metadata = trafilatura.extract_metadata(html)
            title = (metadata.title if metadata and metadata.title else None) or url
            return title, extracted.strip()
    except ImportError:
        logger.warning("trafilatura not installed; falling back to BeautifulSoup extraction")
    except Exception as exc:  # noqa: BLE001
        logger.warning("trafilatura extraction failed for %s: %s", url, exc)

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return title, "\n".join(lines)


def discover_links(html: str, base_url: str, allowed_domain: str) -> list[str]:
    """Return in-domain links found on a page, deduplicated, fragments stripped."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    links: set[str] = set()
    for a in soup.find_all("a", href=True):
        absolute = urljoin(base_url, a["href"])
        absolute, _ = urldefrag(absolute)
        if absolute.startswith("http") and _same_domain(absolute, allowed_domain):
            links.add(absolute)
    return sorted(links)


def load_web_page(url: str, allowed_domain: str, timeout: float = 10.0) -> RawDocument | None:
    """Fetch and extract a single page. Returns None (does not raise) on
    fetch/parse failure so a crawl of many pages can continue past one bad
    page -- callers that need to know about the failure should check logs."""
    if not _same_domain(url, allowed_domain):
        logger.warning("Refusing to load out-of-domain URL during website ingestion: %s", url)
        return None

    try:
        response = fetch_page(url, timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None

    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type:
        logger.info("Skipping non-HTML page %s (content-type=%s)", url, content_type)
        return None

    title, text = extract_text(response.text, url)
    if not text.strip():
        logger.info("No extractable text found on %s", url)
        return None

    document_id = stable_document_id("web", url)
    return RawDocument(
        document_id=document_id,
        source_type="web",
        source_ref=url,
        title=title,
        content=text,
        metadata={
            "source_type": "web",
            "url": url,
            "status_code": response.status_code,
        },
    )


def crawl_website(
    seed_url: str,
    allowed_domain: str,
    max_pages: int = 50,
    timeout: float = 10.0,
) -> list[RawDocument]:
    """Breadth-first crawl of the configured website starting from
    `seed_url`, staying strictly within `allowed_domain`, up to
    `max_pages` pages. Used by ingestion (bulk knowledge load), not by the
    per-request web fallback (which fetches a small, targeted set of pages)."""
    visited: set[str] = set()
    queue: list[str] = [seed_url]
    documents: list[RawDocument] = []

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            response = fetch_page(url, timeout=timeout)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Crawl: failed to fetch %s: %s", url, exc)
            continue

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type:
            continue

        title, text = extract_text(response.text, url)
        if text.strip():
            documents.append(
                RawDocument(
                    document_id=stable_document_id("web", url),
                    source_type="web",
                    source_ref=url,
                    title=title,
                    content=text,
                    metadata={"source_type": "web", "url": url, "status_code": response.status_code},
                )
            )

        for link in discover_links(response.text, url, allowed_domain):
            if link not in visited and link not in queue:
                queue.append(link)

    return documents
