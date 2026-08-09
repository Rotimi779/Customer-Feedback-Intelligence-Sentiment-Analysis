"""Unit tests for dataset summary cards."""

from __future__ import annotations

import pandas as pd
import pytest

from src.eda.summary import build_dataset_summary, format_bytes


def test_dataset_summary_includes_optional_metadata() -> None:
    dataframe = pd.DataFrame(
        {
            "review_text": ["Great application", "Needs better search"],
            "clean_text": ["Great application", "Needs better search"],
            "date": pd.to_datetime(["2026-01-01", "2026-01-05"]),
            "rating": [5, 3],
            "product": ["Mobile", "Web"],
            "category": ["Praise", "Feature"],
        }
    )

    summary = build_dataset_summary(dataframe)

    assert summary.total_reviews == 2
    assert summary.average_review_length == 2.5
    assert summary.median_review_length == 2.5
    assert summary.average_rating == 4.0
    assert summary.minimum_date == pd.Timestamp("2026-01-01")
    assert summary.maximum_date == pd.Timestamp("2026-01-05")
    assert summary.product_count == 2
    assert summary.category_count == 2
    assert summary.dataset_size_bytes > 0


def test_dataset_summary_handles_text_only_data() -> None:
    dataframe = pd.DataFrame(
        {
            "review_text": ["Only review text"],
            "clean_text": ["Only review text"],
        }
    )

    summary = build_dataset_summary(dataframe)

    assert summary.average_rating is None
    assert summary.minimum_date is None
    assert summary.product_count is None
    assert summary.category_count is None


def test_format_bytes() -> None:
    assert format_bytes(512) == "512 B"
    assert format_bytes(1536) == "1.5 KB"
    assert format_bytes(2 * 1024 * 1024) == "2.0 MB"

    with pytest.raises(ValueError):
        format_bytes(-1)
