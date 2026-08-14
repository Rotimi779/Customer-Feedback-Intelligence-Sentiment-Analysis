"""Small, dependency-light text normalization helpers."""

from __future__ import annotations

import re

_WHITESPACE_PATTERN = re.compile(r"\s+")
_REPEATED_PUNCTUATION_PATTERN = re.compile(r"([!?.,])\1{2,}")


def normalize_whitespace(text: object) -> str:
    """Convert a value to text, trim it, and collapse repeated whitespace."""
    if text is None:
        return ""
    return _WHITESPACE_PATTERN.sub(" ", str(text)).strip()


def collapse_repeated_punctuation(text: str) -> str:
    """Reduce runs such as ``!!!!`` to a single punctuation mark."""
    return _REPEATED_PUNCTUATION_PATTERN.sub(r"\1", text)
