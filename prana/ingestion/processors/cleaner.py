"""
Text-level cleanup for raw ingested content.

Responsibilities are strictly surface-level cleanup (whitespace, obvious
extraction artifacts, encoding noise). Structural/semantic decisions
(rejecting empty docs, normalizing metadata) belong in preprocessor.py.
"""

from __future__ import annotations

import re
import unicodedata

_MULTI_BLANK_LINES_RE = re.compile(r"\n{3,}")
_TRAILING_WHITESPACE_RE = re.compile(r"[ \t]+\n")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_PAGE_NUMBER_LINE_RE = re.compile(r"^\s*-?\s*\d{1,4}\s*-?\s*$")
_REPEATED_PUNCT_RE = re.compile(r"([!?.]){3,}")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_unicode(text: str) -> str:
    """NFKC-normalize text to fold visually-identical characters (curly
    quotes, ligatures, full-width punctuation from bad PDF extraction,
    etc.) into a consistent form."""
    return unicodedata.normalize("NFKC", text)


def strip_control_characters(text: str) -> str:
    return _CONTROL_CHARS_RE.sub("", text)


def collapse_whitespace(text: str) -> str:
    text = _TRAILING_WHITESPACE_RE.sub("\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def remove_standalone_page_numbers(text: str) -> str:
    """Drop lines that are just a page number -- a common PDF extraction
    artifact that adds noise without semantic value."""
    lines = text.split("\n")
    kept = [line for line in lines if not _PAGE_NUMBER_LINE_RE.match(line)]
    return "\n".join(kept)


def normalize_repeated_punctuation(text: str) -> str:
    """Collapse runs like '......' or '!!!!' down to a single mark; these
    are almost always extraction/formatting artifacts, not intentional
    emphasis worth preserving for retrieval."""
    return _REPEATED_PUNCT_RE.sub(r"\1", text)


def clean_text(text: str) -> str:
    """Apply the full cleanup pipeline in a safe order.

    Deliberately conservative: this must never remove content that changes
    meaning. Each step targets a specific, well-understood artifact class.
    """
    if not text:
        return ""

    text = normalize_unicode(text)
    text = strip_control_characters(text)
    text = remove_standalone_page_numbers(text)
    text = normalize_repeated_punctuation(text)
    text = collapse_whitespace(text)
    return text
