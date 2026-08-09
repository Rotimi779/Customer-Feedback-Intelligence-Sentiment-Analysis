"""Descriptive statistics for customer-feedback datasets."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Final, Iterable

import pandas as pd


_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z][A-Za-z']+")
DEFAULT_STOP_WORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "about",
        "after",
        "again",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "for",
        "from",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "me",
        "more",
        "most",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "out",
        "over",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "too",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)


@dataclass(frozen=True, slots=True)
class ReviewLengthStatistics:
    """Word- and character-length statistics for review text."""

    average_words: float
    median_words: float
    minimum_words: int
    maximum_words: int
    percentile_90_words: float
    average_characters: float


@dataclass(frozen=True, slots=True)
class RatingStatistics:
    """Summary statistics for an optional numeric rating column."""

    count: int
    average: float
    median: float
    minimum: float
    maximum: float


def add_review_length_columns(
    dataframe: pd.DataFrame,
    *,
    text_column: str = "review_text",
) -> pd.DataFrame:
    """Return a copy containing word and character length columns."""
    result = dataframe.copy()
    if text_column not in result.columns:
        result["review_word_count"] = pd.Series(0, index=result.index, dtype="int64")
        result["review_character_count"] = pd.Series(
            0,
            index=result.index,
            dtype="int64",
        )
        return result

    text = result[text_column].astype("string").fillna("").str.strip()
    result["review_word_count"] = text.str.findall(r"\b\w+\b").str.len().astype("int64")
    result["review_character_count"] = text.str.len().astype("int64")
    return result


def calculate_review_length_statistics(
    dataframe: pd.DataFrame,
    *,
    text_column: str = "review_text",
) -> ReviewLengthStatistics:
    """Calculate review-length statistics, returning zeros for empty input."""
    enriched = add_review_length_columns(
        dataframe,
        text_column=text_column,
    )
    lengths = enriched["review_word_count"]

    if lengths.empty:
        return ReviewLengthStatistics(0.0, 0.0, 0, 0, 0.0, 0.0)

    character_lengths = enriched["review_character_count"]
    return ReviewLengthStatistics(
        average_words=float(lengths.mean()),
        median_words=float(lengths.median()),
        minimum_words=int(lengths.min()),
        maximum_words=int(lengths.max()),
        percentile_90_words=float(lengths.quantile(0.90)),
        average_characters=float(character_lengths.mean()),
    )


def calculate_rating_statistics(
    dataframe: pd.DataFrame,
    *,
    rating_column: str = "rating",
) -> RatingStatistics | None:
    """Calculate numeric rating statistics when a usable column exists."""
    if rating_column not in dataframe.columns:
        return None

    ratings = pd.to_numeric(dataframe[rating_column], errors="coerce").dropna()
    if ratings.empty:
        return None

    return RatingStatistics(
        count=int(ratings.count()),
        average=float(ratings.mean()),
        median=float(ratings.median()),
        minimum=float(ratings.min()),
        maximum=float(ratings.max()),
    )


def summarize_reviews_over_time(
    dataframe: pd.DataFrame,
    *,
    date_column: str = "date",
    frequency: str = "D",
) -> pd.DataFrame:
    """Aggregate review counts into deterministic time buckets."""
    if date_column not in dataframe.columns:
        return pd.DataFrame(columns=("period", "review_count"))

    dates = pd.to_datetime(dataframe[date_column], errors="coerce").dropna()
    if dates.empty:
        return pd.DataFrame(columns=("period", "review_count"))

    periods = dates.dt.to_period(frequency).dt.to_timestamp()
    counts = periods.value_counts().sort_index()
    return counts.rename_axis("period").reset_index(name="review_count")


def infer_time_frequency(
    dataframe: pd.DataFrame,
    *,
    date_column: str = "date",
) -> str:
    """Choose daily, weekly, or monthly buckets based on the observed span."""
    if date_column not in dataframe.columns:
        return "D"

    dates = pd.to_datetime(dataframe[date_column], errors="coerce").dropna()
    if dates.empty:
        return "D"

    span_days = int((dates.max() - dates.min()).days)
    if span_days <= 90:
        return "D"
    if span_days <= 730:
        return "W"
    return "M"


def _iter_tokens(
    values: Iterable[object],
    *,
    stop_words: frozenset[str],
    minimum_token_length: int,
) -> Iterable[str]:
    for value in values:
        if pd.isna(value):
            continue
        for token in _TOKEN_PATTERN.findall(str(value).casefold()):
            normalized = token.strip("'")
            if (
                len(normalized) >= minimum_token_length
                and normalized not in stop_words
            ):
                yield normalized


def calculate_common_words(
    dataframe: pd.DataFrame,
    *,
    text_column: str = "clean_text",
    top_n: int = 20,
    minimum_token_length: int = 2,
    stop_words: frozenset[str] = DEFAULT_STOP_WORDS,
) -> pd.DataFrame:
    """Return frequent terms after lightweight, explainable preprocessing."""
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero.")
    if minimum_token_length <= 0:
        raise ValueError("minimum_token_length must be greater than zero.")
    if text_column not in dataframe.columns:
        return pd.DataFrame(columns=("term", "count", "percentage"))

    counter = Counter(
        _iter_tokens(
            dataframe[text_column].tolist(),
            stop_words=stop_words,
            minimum_token_length=minimum_token_length,
        )
    )
    total_tokens = sum(counter.values())
    records = [
        {
            "term": term,
            "count": count,
            "percentage": 0.0 if total_tokens == 0 else (count / total_tokens) * 100.0,
        }
        for term, count in counter.most_common(top_n)
    ]
    return pd.DataFrame(records, columns=("term", "count", "percentage"))
