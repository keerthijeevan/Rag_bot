"""
Load PDF files into RawDocument objects.

Uses `pypdf` (maintained, pure-Python) rather than a hand-rolled PDF parser.
"""

from __future__ import annotations

import logging
import os

from ingestion.schema import RawDocument, stable_document_id

logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> RawDocument:
    """Load a single PDF file and return one RawDocument per file.

    Page boundaries are preserved as `\\f` (form feed) so downstream
    chunkers can still find them if useful, and per-page text is also kept
    in metadata for source attribution (page numbers).
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "pypdf is required for PDF ingestion. Install it with `pip install pypdf`."
        ) from exc

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    reader = PdfReader(file_path)

    page_texts: list[str] = []
    for page in reader.pages:
        try:
            page_texts.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001 - a single bad page shouldn't kill ingestion
            logger.warning("Failed to extract text from a page of %s: %s", file_path, exc)
            page_texts.append("")

    full_text = "\f".join(page_texts)

    doc_info = reader.metadata or {}
    title = (doc_info.get("/Title") or os.path.basename(file_path)).strip()

    document_id = stable_document_id("pdf", file_path)

    return RawDocument(
        document_id=document_id,
        source_type="pdf",
        source_ref=file_path,
        title=title,
        content=full_text,
        metadata={
            "source_type": "pdf",
            "file_name": os.path.basename(file_path),
            "page_count": len(reader.pages),
            "page_texts": page_texts,  # used later for accurate page attribution during chunking
            "author": doc_info.get("/Author"),
        },
    )


def load_pdfs_from_dir(directory: str) -> list[RawDocument]:
    """Load every .pdf file directly inside `directory`."""
    if not os.path.isdir(directory):
        return []
    documents: list[RawDocument] = []
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith(".pdf"):
            path = os.path.join(directory, name)
            try:
                documents.append(load_pdf(path))
            except Exception as exc:  # noqa: BLE001
                logger.error("Skipping unreadable PDF %s: %s", path, exc)
    return documents
