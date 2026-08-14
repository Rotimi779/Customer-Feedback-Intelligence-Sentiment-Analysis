"""Sentiment-specific preprocessing entry points.

Training and inference import these functions so both paths always apply the same
text transformations.
"""

from __future__ import annotations

from collections.abc import Iterable

from src.preprocessing import clean_for_classical_model, clean_for_transformer


def prepare_classical_text(text: object) -> str:
    """Return one TF-IDF-ready string."""
    return clean_for_classical_model(text)


def prepare_classical_texts(texts: Iterable[object]) -> list[str]:
    """Return TF-IDF-ready strings in input order."""
    return [prepare_classical_text(text) for text in texts]


def prepare_transformer_text(text: object) -> str:
    """Return minimally cleaned DistilBERT input."""
    return clean_for_transformer(text)


def prepare_transformer_texts(texts: Iterable[object]) -> list[str]:
    """Return minimally cleaned DistilBERT inputs in input order."""
    return [prepare_transformer_text(text) for text in texts]
