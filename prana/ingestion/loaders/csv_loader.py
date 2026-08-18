"""
Load CSV files into RawDocument objects.

Each row is converted into a small "field: value" record so it reads as
natural text for embedding, rather than being a comma-separated blob. Each
row becomes one RawDocument so metadata filtering (e.g. by document_id) can
target individual rows if needed; a combined document is not created since
that would blur row-level source attribution.
"""

from __future__ import annotations

import csv
import logging
import os

from ingestion.schema import RawDocument, stable_document_id

logger = logging.getLogger(__name__)


def _row_to_text(headers: list[str], row: list[str]) -> str:
    parts = []
    for header, value in zip(headers, row):
        value = (value or "").strip()
        if value:
            parts.append(f"{header.strip()}: {value}")
    return "\n".join(parts)


def load_csv(file_path: str, title_column: str | None = None) -> list[RawDocument]:
    """Load a CSV file, returning one RawDocument per row.

    Args:
        file_path: path to the .csv file.
        title_column: optional column name to use as each row's title
            (falls back to "<file> row <n>" when not provided or missing).
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"CSV not found: {file_path}")

    documents: list[RawDocument] = []
    file_name = os.path.basename(file_path)

    with open(file_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            return []

        for row_index, row in enumerate(reader):
            if not any(cell.strip() for cell in row if cell):
                continue  # skip fully-empty rows

            text = _row_to_text(headers, row)
            if not text.strip():
                continue

            title = None
            if title_column and title_column in headers:
                col_idx = headers.index(title_column)
                if col_idx < len(row):
                    title = row[col_idx].strip() or None
            if not title:
                title = f"{file_name} row {row_index + 1}"

            source_ref = f"{file_path}#row={row_index + 1}"
            document_id = stable_document_id("csv", source_ref)

            documents.append(
                RawDocument(
                    document_id=document_id,
                    source_type="csv",
                    source_ref=source_ref,
                    title=title,
                    content=text,
                    metadata={
                        "source_type": "csv",
                        "file_name": file_name,
                        "row_index": row_index + 1,
                    },
                )
            )

    return documents


def load_csv_from_dir(directory: str) -> list[RawDocument]:
    if not os.path.isdir(directory):
        return []
    documents: list[RawDocument] = []
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith(".csv"):
            path = os.path.join(directory, name)
            try:
                documents.extend(load_csv(path))
            except Exception as exc:  # noqa: BLE001
                logger.error("Skipping unreadable CSV %s: %s", path, exc)
    return documents
