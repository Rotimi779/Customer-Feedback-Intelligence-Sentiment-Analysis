"""Tests for reusable Plotly EDA chart builders."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.eda.visualizations import (
    build_common_words_chart,
    build_group_distribution_chart,
    build_missing_values_chart,
    build_rating_distribution_chart,
    build_review_length_histogram,
    build_reviews_over_time_chart,
)


def _full_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_text": ["Fast helpful app", "Slow update"],
            "clean_text": ["Fast helpful app", "Slow update"],
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "rating": [5, 2],
            "product": ["Mobile", "Mobile"],
            "category": ["Praise", None],
        }
    )


def test_required_charts_return_plotly_figures() -> None:
    dataframe = _full_dataset()

    assert isinstance(build_review_length_histogram(dataframe), go.Figure)
    assert isinstance(build_common_words_chart(dataframe), go.Figure)
    assert isinstance(build_missing_values_chart(dataframe), go.Figure)


def test_optional_charts_hide_when_columns_are_unavailable() -> None:
    dataframe = pd.DataFrame(
        {
            "review_text": ["Text only review"],
            "clean_text": ["Text only review"],
        }
    )

    assert build_reviews_over_time_chart(dataframe) is None
    assert build_rating_distribution_chart(dataframe) is None
    assert build_group_distribution_chart(dataframe, column="product") is None


def test_optional_charts_render_when_metadata_exists() -> None:
    dataframe = _full_dataset()

    assert isinstance(build_reviews_over_time_chart(dataframe), go.Figure)
    assert isinstance(build_rating_distribution_chart(dataframe), go.Figure)
    assert isinstance(
        build_group_distribution_chart(dataframe, column="product"),
        go.Figure,
    )
