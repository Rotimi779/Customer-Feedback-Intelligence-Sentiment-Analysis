"""Unit tests for shared dashboard filters."""

from __future__ import annotations

import pandas as pd
import pytest

from src.eda.filters import DatasetFilters, apply_filters, get_filter_options


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": ["A", "B", "C", "D"],
            "review_text": [
                "Fast mobile experience",
                "Slow mobile update",
                "Helpful web support",
                "Billing page failed",
            ],
            "clean_text": [
                "Fast mobile experience",
                "Slow mobile update",
                "Helpful web support",
                "Billing page failed",
            ],
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-10", "2026-02-01", "2026-02-15"]
            ),
            "rating": [5, 2, 4, 1],
            "product": ["Mobile", "Mobile", "Web", "Web"],
            "category": ["Performance", "Performance", "Support", "Billing"],
        }
    )


def test_filter_options_reflect_available_metadata() -> None:
    options = get_filter_options(_dataset())

    assert options.minimum_date == pd.Timestamp("2026-01-01")
    assert options.maximum_date == pd.Timestamp("2026-02-15")
    assert options.products == ("Mobile", "Web")
    assert options.categories == ("Billing", "Performance", "Support")
    assert options.minimum_rating == 1.0
    assert options.maximum_rating == 5.0


def test_filters_are_combined_and_non_mutating() -> None:
    dataframe = _dataset()
    original = dataframe.copy(deep=True)

    filtered = apply_filters(
        dataframe,
        DatasetFilters(
            start_date="2026-01-01",
            end_date="2026-01-31",
            products=("Mobile",),
            rating_min=3,
            keyword="fast",
        ),
    )

    assert filtered["review_id"].tolist() == ["A"]
    pd.testing.assert_frame_equal(dataframe, original)


def test_filters_skip_missing_optional_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "review_text": ["Good app", "Bad app"],
            "clean_text": ["Good app", "Bad app"],
        }
    )

    filtered = apply_filters(
        dataframe,
        DatasetFilters(products=("Mobile",), rating_min=5),
    )

    pd.testing.assert_frame_equal(filtered, dataframe)


def test_keyword_search_is_literal_and_case_insensitive() -> None:
    dataframe = pd.DataFrame(
        {
            "review_text": ["Search [works]", "SEARCH works", "No match"],
            "clean_text": ["Search [works]", "SEARCH works", "No match"],
        }
    )

    filtered = apply_filters(dataframe, DatasetFilters(keyword="[WORKS]"))

    assert filtered["review_text"].tolist() == ["Search [works]"]


def test_invalid_date_filter_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid date"):
        apply_filters(_dataset(), DatasetFilters(start_date="not-a-date"))
