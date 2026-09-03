"""Shared constants and helpers for rule-based aspect analysis."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


SENTIMENT_TO_SCORE: dict[str, float] = {
    "Negative": -1.0,
    "Neutral": 0.0,
    "Positive": 1.0,
}

# The MVP intentionally uses a compact, transparent, domain-general vocabulary.
# Callers may inject a different mapping for a specific domain without changing
# the extraction algorithm.
DEFAULT_ASPECT_VOCABULARY: dict[str, tuple[str, ...]] = {
    "Battery": (
        "battery",
        "battery life",
        "charging",
        "charger",
    ),
    "Customer Support": (
        "customer support",
        "customer service",
        "help desk",
        "support",
        "support agent",
        "agent",
        "representative",
    ),
    "User Interface": (
        "user interface",
        "interface",
        "ui",
        "navigation",
        "menu",
        "layout",
    ),
    "Shipping": (
        "shipping",
        "shipment",
        "delivery",
        "delivered",
        "tracking",
        "courier",
    ),
    "Packaging": (
        "packaging",
        "packaged",
        "package",
        "parcel",
        "box",
    ),
    "Billing": (
        "billing",
        "billed",
        "charged",
        "refund",
        "payment",
        "invoice",
        "subscription",
    ),
    "Price": (
        "price",
        "pricing",
        "cost",
        "expensive",
        "cheap",
        "affordable",
        "overpriced",
        "value for money",
    ),
    "Performance": (
        "performance",
        "slow",
        "slower",
        "lag",
        "laggy",
        "speed",
        "loading",
        "load time",
        "crash",
        "crashes",
        "crashing",
        "freeze",
        "freezes",
        "freezing",
        "latency",
    ),
    "Quality": (
        "quality",
        "build quality",
        "durable",
        "durability",
        "broken",
        "defective",
        "material",
    ),
    "Ease of Use": (
        "ease of use",
        "easy to use",
        "hard to use",
        "difficult to use",
        "usability",
        "intuitive",
        "confusing",
        "setup",
        "installation",
        "instructions",
    ),
    "Features": (
        "feature",
        "features",
        "functionality",
    ),
    "Reliability": (
        "reliable",
        "unreliable",
        "reliability",
        "stable",
        "unstable",
        "stability",
    ),
}


class AspectAnalysisError(ValueError):
    """Raised when aspect analysis cannot be completed safely."""


@dataclass(frozen=True)
class AspectMatch:
    """One canonical aspect and the source terms that triggered it."""

    aspect: str
    matched_terms: tuple[str, ...]


def normalize_aspect_text(text: object) -> str:
    """Normalize text for deterministic keyword and phrase matching."""
    value = "" if text is None else str(text)
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9']+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def validate_aspect_vocabulary(
    vocabulary: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    """Validate and normalize an injected aspect vocabulary."""
    if not vocabulary:
        raise AspectAnalysisError("Aspect vocabulary cannot be empty.")

    normalized: dict[str, tuple[str, ...]] = {}
    for raw_aspect, raw_terms in vocabulary.items():
        aspect = str(raw_aspect).strip()
        if not aspect:
            raise AspectAnalysisError("Aspect names cannot be empty.")

        terms: list[str] = []
        for raw_term in raw_terms:
            term = normalize_aspect_text(raw_term)
            if term and term not in terms:
                terms.append(term)
        if not terms:
            raise AspectAnalysisError(f"Aspect '{aspect}' must define at least one term.")
        normalized[aspect] = tuple(terms)

    return normalized
