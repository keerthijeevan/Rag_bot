from __future__ import annotations

import os
import tempfile

from ingestion.loaders.csv_loader import load_csv
from ingestion.loaders.markdown_loader import load_markdown
from ingestion.processors.cleaner import (
    clean_text,
    collapse_whitespace,
    normalize_repeated_punctuation,
    remove_standalone_page_numbers,
)


def test_collapse_whitespace_reduces_blank_lines():
    text = "para one\n\n\n\n\npara two"
    result = collapse_whitespace(text)
    assert "\n\n\n" not in result


def test_remove_standalone_page_numbers():
    text = "Real content line.\n42\nMore real content."
    result = remove_standalone_page_numbers(text)
    assert "42" not in result.split("\n")
    assert "Real content line." in result


def test_normalize_repeated_punctuation():
    assert normalize_repeated_punctuation("Wow!!!!") == "Wow!"
    assert normalize_repeated_punctuation("Wait......") == "Wait."


def test_clean_text_handles_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""  # type: ignore[arg-type]


def test_clean_text_preserves_semantic_content():
    text = "Our refund policy allows returns within 30 days."
    result = clean_text(text)
    assert "refund policy" in result
    assert "30 days" in result


def test_load_markdown_extracts_title_from_first_heading():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "doc.md")
        with open(path, "w") as f:
            f.write("# Getting Started\n\nSome content here about the product.\n")
        doc = load_markdown(path)
        assert doc.title == "Getting Started"
        assert doc.source_type == "markdown"
        assert "Some content" in doc.content


def test_load_csv_produces_one_document_per_row():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "faq.csv")
        with open(path, "w", newline="") as f:
            f.write("question,answer\n")
            f.write("What is your refund policy?,Refunds within 30 days.\n")
            f.write("Do you ship internationally?,Yes to most countries.\n")

        docs = load_csv(path, title_column="question")
        assert len(docs) == 2
        assert docs[0].title == "What is your refund policy?"
        assert "Refunds within 30 days." in docs[0].content
        assert docs[0].source_type == "csv"


def test_load_csv_skips_fully_empty_rows():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "data.csv")
        with open(path, "w", newline="") as f:
            f.write("a,b\n")
            f.write("1,2\n")
            f.write(",\n")  # empty row
            f.write("3,4\n")

        docs = load_csv(path)
        assert len(docs) == 2
