"""
Load plain-text (.txt) files into RawDocument objects.
"""

from __future__ import annotations

import logging
import os

from ingestion.schema import RawDocument, stable_document_id

logger = logging.getLogger(__name__)


def load_txt(file_path: str) -> RawDocument:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"TXT not found: {file_path}")

    # Try a couple of common encodings before giving up, since ingested
    # text files are not guaranteed to be UTF-8.
    encodings = ("utf-8", "utf-8-sig", "latin-1")
    content: str | None = None
    used_encoding = None
    for enc in encodings:
        try:
            with open(file_path, encoding=enc) as f:
                content = f.read()
            used_encoding = enc
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        raise ValueError(f"Could not decode {file_path} with any of {encodings}")

    title = os.path.splitext(os.path.basename(file_path))[0]
    document_id = stable_document_id("txt", file_path)

    return RawDocument(
        document_id=document_id,
        source_type="txt",
        source_ref=file_path,
        title=title,
        content=content,
        metadata={
            "source_type": "txt",
            "file_name": os.path.basename(file_path),
            "encoding": used_encoding,
        },
    )


def load_txt_from_dir(directory: str) -> list[RawDocument]:
    if not os.path.isdir(directory):
        return []
    documents: list[RawDocument] = []
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith(".txt"):
            path = os.path.join(directory, name)
            try:
                documents.append(load_txt(path))
            except Exception as exc:  # noqa: BLE001
                logger.error("Skipping unreadable TXT %s: %s", path, exc)
    return documents
