"""Shared types and evidence helpers for deterministic business insights."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any, Iterable

import pandas as pd


SENTIMENT_TO_SCORE: dict[str, float] = {
    "Negative": -1.0,
    "Neutral": 0.0,
    "Positive": 1.0,
}

PRIORITY_ORDER: dict[str, int] = {"High": 0, "Medium": 1, "Low": 2}


class InsightsError(ValueError):
    """Raised when business insights cannot be generated safely."""


@dataclass(frozen=True)
class InsightFinding:
    """One evidence-backed finding shown on the Insights page."""

    category: str
    title: str
    evidence: str
    business_interpretation: str
    metric_name: str
    metric_value: float | str
    supporting_count: int
    affected_item: str | None = None
    representative_review_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a DataFrame/JSON-friendly representation."""
        payload = asdict(self)
        payload["representative_review_ids"] = list(self.representative_review_ids)
        return payload


@dataclass(frozen=True)
class InsightRecommendation:
    """One cautious rule-based recommendation with explicit evidence."""

    priority: str
    title: str
    affected_type: str
    affected_item: str
    supporting_metric: str
    supporting_value: float
    supporting_count: int
    explanation: str
    representative_review_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a DataFrame/JSON-friendly representation."""
        payload = asdict(self)
        payload["representative_review_ids"] = list(self.representative_review_ids)
        return payload


def canonical_sentiment_series(values: pd.Series) -> pd.Series:
    """Normalize sentiment labels and reject unsupported values."""
    normalized = values.astype("string").str.strip().str.title()
    invalid = normalized.notna() & ~normalized.isin(SENTIMENT_TO_SCORE)
    if bool(invalid.any()):
        bad = sorted(normalized.loc[invalid].dropna().astype(str).unique())
        raise InsightsError(
            "Business insights require canonical sentiment labels: Negative, Neutral, "
            f"or Positive. Unsupported labels: {', '.join(bad)}"
        )
    return normalized


def minimum_evidence_support(total_reviews: int) -> int:
    """Return a conservative support floor for ranked aspect/topic claims."""
    if total_reviews <= 0:
        return 1
    return max(2, int(ceil(total_reviews * 0.02)))


def representative_ids(
    dataframe: pd.DataFrame,
    *,
    mask: pd.Series | None = None,
    limit: int = 3,
) -> tuple[str, ...]:
    """Return a small deterministic set of review IDs supporting a claim."""
    if limit <= 0 or "review_id" not in dataframe.columns or dataframe.empty:
        return ()
    selected = dataframe if mask is None else dataframe.loc[mask]
    if selected.empty:
        return ()
    return tuple(selected["review_id"].astype(str).drop_duplicates().head(limit).tolist())


def coerce_review_ids(value: object) -> tuple[str, ...]:
    """Normalize a stored review-ID collection for display/export."""
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    return ()
