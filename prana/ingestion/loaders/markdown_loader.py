"""
Load Markdown (.md) files into RawDocument objects.

We keep the raw Markdown (headings, lists, etc.) intact in `content` rather
than stripping it -- structure like headings is useful downstream for
semantic chunking and for readable source attribution. A lightly-rendered
plain-text version is also stored in metadata for cases where clean prose
is preferred.
"""

from __future__ import annotations

import logging
import os
import re

from ingestion.schema import RawDocument, stable_document_id

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _first_heading(text: str) -> str | None:
    match = _HEADING_RE.search(text)
    return match.group(2).strip() if match else None


def _strip_markdown_syntax(text: str) -> str:
    """Very light markdown -> plain text conversion for metadata only."""
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)  # inline/code blocks (rough)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links -> text
    text = re.sub(r"[#*_>-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_markdown(file_path: str) -> RawDocument:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Markdown file not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        raw = f.read()

    title = _first_heading(raw) or os.path.splitext(os.path.basename(file_path))[0]
    document_id = stable_document_id("markdown", file_path)

    return RawDocument(
        document_id=document_id,
        source_type="markdown",
        source_ref=file_path,
        title=title,
        content=raw,
        metadata={
            "source_type": "markdown",
            "file_name": os.path.basename(file_path),
            "plain_text_preview": _strip_markdown_syntax(raw)[:500],
        },
    )


def load_markdown_from_dir(directory: str) -> list[RawDocument]:
    if not os.path.isdir(directory):
        return []
    documents: list[RawDocument] = []
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith((".md", ".markdown")):
            path = os.path.join(directory, name)
            try:
                documents.append(load_markdown(path))
            except Exception as exc:  # noqa: BLE001
                logger.error("Skipping unreadable Markdown file %s: %s", path, exc)
    return documents
