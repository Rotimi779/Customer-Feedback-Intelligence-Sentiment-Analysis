"""Keyword extraction and concise topic-label generation."""

from __future__ import annotations

import numpy as np


def extract_topic_keywords(
    components: np.ndarray,
    feature_names: np.ndarray,
    *,
    top_n: int,
) -> dict[int, list[str]]:
    """Return highest-weighted TF-IDF features for each NMF component."""
    keywords: dict[int, list[str]] = {}
    for topic_id, weights in enumerate(components):
        top_indices = np.argsort(weights)[::-1][:top_n]
        keywords[topic_id] = [str(feature_names[index]) for index in top_indices]
    return keywords


def _select_label_terms(keywords: list[str], max_terms: int) -> list[str]:
    """Keep highest-weighted distinct terms while avoiding obvious redundancy."""
    selected: list[str] = []

    for term in keywords:
        normalized = term.strip().lower()
        if not normalized:
            continue
        if any(
            normalized == existing
            or normalized in existing.split()
            or existing in normalized.split()
            for existing in (item.lower() for item in selected)
        ):
            continue
        selected.append(term)
        if len(selected) >= max_terms:
            break

    return selected or keywords[:max_terms]


def generate_topic_label(keywords: list[str], *, max_terms: int = 2) -> str:
    """Create a transparent human-readable label from top weighted keywords."""
    if not keywords:
        return "Unlabelled Topic"
    terms = _select_label_terms(keywords, max_terms)
    return " / ".join(term.title() for term in terms)


def generate_topic_labels(
    topic_keywords: dict[int, list[str]],
    *,
    max_terms: int = 2,
) -> dict[int, str]:
    """Create one readable label for every discovered topic."""
    return {
        topic_id: generate_topic_label(keywords, max_terms=max_terms)
        for topic_id, keywords in topic_keywords.items()
    }
