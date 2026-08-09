"""Dataset-level summaries used by the Overview dashboard."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.eda.quality import count_duplicate_reviews, count_empty_reviews
from src.eda.statistics import (
    calculate_rating_statistics,
    calculate_review_length_statistics,
)


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """Core EDA metrics for the current, possibly filtered dataset."""

    total_reviews: int
    duplicate_reviews: int
    missing_reviews: int
    average_review_length: float
    median_review_length: float
    dataset_size_bytes: int
    average_rating: float | None
    minimum_date: pd.Timestamp | None
    maximum_date: pd.Timestamp | None
    product_count: int | None
    category_count: int | None


def _date_range(
    dataframe: pd.DataFrame,
    *,
    date_column: str,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if date_column not in dataframe.columns:
        return None, None
    dates = pd.to_datetime(dataframe[date_column], errors="coerce").dropna()
    if dates.empty:
        return None, None
    return pd.Timestamp(dates.min()), pd.Timestamp(dates.max())


def _unique_count(dataframe: pd.DataFrame, column: str) -> int | None:
    if column not in dataframe.columns:
        return None
    return int(dataframe[column].dropna().nunique())


def build_dataset_summary(
    dataframe: pd.DataFrame,
    *,
    text_column: str = "review_text",
    duplicate_text_column: str = "clean_text",
    date_column: str = "date",
    rating_column: str = "rating",
    product_column: str = "product",
    category_column: str = "category",
) -> DatasetSummary:
    """Build the minimum required EDA metrics without mutating input data."""
    lengths = calculate_review_length_statistics(
        dataframe,
        text_column=text_column,
    )
    ratings = calculate_rating_statistics(
        dataframe,
        rating_column=rating_column,
    )
    minimum_date, maximum_date = _date_range(
        dataframe,
        date_column=date_column,
    )

    duplicate_column = (
        duplicate_text_column
        if duplicate_text_column in dataframe.columns
        else text_column
    )
    return DatasetSummary(
        total_reviews=len(dataframe),
        duplicate_reviews=count_duplicate_reviews(
            dataframe,
            text_column=duplicate_column,
        ),
        missing_reviews=count_empty_reviews(
            dataframe,
            text_column=text_column,
        ),
        average_review_length=lengths.average_words,
        median_review_length=lengths.median_words,
        dataset_size_bytes=int(dataframe.memory_usage(index=True, deep=True).sum()),
        average_rating=None if ratings is None else ratings.average,
        minimum_date=minimum_date,
        maximum_date=maximum_date,
        product_count=_unique_count(dataframe, product_column),
        category_count=_unique_count(dataframe, category_column),
    )


def format_bytes(size_bytes: int) -> str:
    """Format a byte count for compact KPI display."""
    if size_bytes < 0:
        raise ValueError("size_bytes cannot be negative.")

    value = float(size_bytes)
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            precision = 0 if unit == "B" else 1
            return f"{value:.{precision}f} {unit}"
        value /= 1024.0
    raise RuntimeError("Unable to format byte value.")
