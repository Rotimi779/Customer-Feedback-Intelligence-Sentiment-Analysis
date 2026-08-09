"""Unit tests for EDA quality metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from src.eda.quality import (
    count_duplicate_reviews,
    count_empty_reviews,
    missing_value_summary,
    summarize_quality,
)


def test_missing_summary_counts_nulls_and_blank_strings() -> None:
    dataframe = pd.DataFrame(
        {
            "review_text": ["Useful", "  ", None],
            "rating": [5, None, 2],
        }
    )

    summary = missing_value_summary(dataframe).set_index("column")

    assert summary.loc["review_text", "missing_count"] == 2
    assert summary.loc["rating", "missing_count"] == 1
    assert summary.loc["review_text", "missing_percentage"] == pytest.approx(200 / 3)


def test_duplicate_detection_normalizes_case_and_whitespace() -> None:
    dataframe = pd.DataFrame(
        {
            "clean_text": [
                "Great product",
                "  great   PRODUCT ",
                "Different review",
                "",
            ]
        }
    )

    assert count_duplicate_reviews(dataframe) == 1


def test_empty_review_detection_is_safe_when_column_is_absent() -> None:
    dataframe = pd.DataFrame({"rating": [1, 2]})

    assert count_empty_reviews(dataframe) == 0


def test_quality_summary_calculates_completeness() -> None:
    dataframe = pd.DataFrame(
        {
            "clean_text": ["First review", "Second review"],
            "product": ["App", None],
        }
    )

    summary = summarize_quality(dataframe)

    assert summary.total_rows == 2
    assert summary.missing_cells == 1
    assert summary.total_cells == 4
    assert summary.completeness_percentage == 75.0
