"""Shared topic-modeling configuration and summary helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd


SENTIMENT_TO_SCORE = {
    "Negative": -1.0,
    "Neutral": 0.0,
    "Positive": 1.0,
}


class TopicModelError(ValueError):
    """Raised when topic modeling cannot be completed safely."""


@dataclass(frozen=True)
class TopicModelConfig:
    """Configuration for the MVP NMF topic model."""

    n_topics: int = 8
    max_features: int = 5000
    min_df: int = 2
    max_df: float = 0.95
    ngram_range: tuple[int, int] = (1, 2)
    top_n_words: int = 8
    random_state: int = 42
    max_iter: int = 400

    def __post_init__(self) -> None:
        if self.n_topics < 2:
            raise ValueError("n_topics must be at least 2.")
        if self.max_features < self.n_topics:
            raise ValueError("max_features must be at least n_topics.")
        if self.min_df < 1:
            raise ValueError("min_df must be at least 1.")
        if not 0 < self.max_df <= 1:
            raise ValueError("max_df must be in the interval (0, 1].")
        if self.top_n_words < 1:
            raise ValueError("top_n_words must be positive.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive.")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""
        payload = asdict(self)
        payload["ngram_range"] = list(self.ngram_range)
        return payload


def choose_topic_text_column(dataframe: pd.DataFrame) -> str:
    """Prefer canonical clean text while allowing labelled public datasets."""
    for candidate in ("clean_text", "review_text"):
        if candidate in dataframe.columns:
            return candidate
    raise TopicModelError(
        "Topic modeling requires a 'clean_text' or 'review_text' column."
    )


def build_topic_summary(
    dataframe: pd.DataFrame,
    *,
    topic_keywords: dict[int, list[str]],
    topic_labels: dict[int, str],
) -> pd.DataFrame:
    """Aggregate topic frequency and sentiment into one reusable table."""
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "topic_id",
                "topic_label",
                "top_keywords",
                "review_count",
                "percentage",
                "average_sentiment",
                "dominant_sentiment",
            ]
        )

    rows: list[dict[str, object]] = []
    total = len(dataframe)

    for topic_id in sorted(topic_labels):
        subset = dataframe.loc[dataframe["topic_id"].eq(topic_id)]
        if subset.empty:
            continue

        average_sentiment: float | None = None
        dominant_sentiment: str | None = None
        if "sentiment_label" in subset.columns:
            standardized = subset["sentiment_label"].astype(str).str.title()
            numeric = standardized.map(SENTIMENT_TO_SCORE)
            if numeric.notna().any():
                average_sentiment = float(numeric.dropna().mean())
            counts = standardized.value_counts()
            if not counts.empty:
                dominant_sentiment = str(counts.index[0])

        rows.append(
            {
                "topic_id": int(topic_id),
                "topic_label": topic_labels[topic_id],
                "top_keywords": ", ".join(topic_keywords[topic_id]),
                "review_count": int(len(subset)),
                "percentage": float(len(subset) / total),
                "average_sentiment": average_sentiment,
                "dominant_sentiment": dominant_sentiment,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["review_count", "topic_id"], ascending=[False, True]
    ).reset_index(drop=True)


def topic_assignment_coverage(assignments: Iterable[int], total_rows: int) -> float:
    """Return the share of rows with non-negative topic assignments."""
    if total_rows == 0:
        return 0.0
    valid = sum(1 for assignment in assignments if int(assignment) >= 0)
    return float(valid / total_rows)
