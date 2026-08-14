"""Model-aware text cleaning used by sentiment training and inference."""

from __future__ import annotations

from src.preprocessing.text_utils import (
    collapse_repeated_punctuation,
    normalize_whitespace,
)


def clean_for_classical_model(text: object) -> str:
    """Prepare text for TF-IDF while preserving negation words.

    The classical path lowercases text and reduces excessive punctuation, but it
    intentionally does not remove stop words because doing so can remove useful
    negation such as ``not`` and ``never``.
    """
    normalized = normalize_whitespace(text).lower()
    return collapse_repeated_punctuation(normalized)


def clean_for_transformer(text: object) -> str:
    """Minimally normalize transformer input without altering casing/punctuation."""
    return normalize_whitespace(text)
