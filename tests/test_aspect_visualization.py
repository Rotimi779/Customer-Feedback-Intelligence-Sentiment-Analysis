"""Tests for reusable aspect-analysis chart builders."""

import pandas as pd
import plotly.graph_objects as go

from src.aspects.aggregation import analyze_aspects
from src.aspects.visualization import (
    build_aspect_frequency_chart,
    build_aspect_rating_chart,
    build_aspect_sentiment_chart,
    build_positive_negative_chart,
)


def test_aspect_visualizations_build_for_nonempty_results(
    aspect_topic_df: pd.DataFrame,
) -> None:
    result = analyze_aspects(aspect_topic_df)

    assert isinstance(build_aspect_frequency_chart(result.summary), go.Figure)
    assert isinstance(build_aspect_sentiment_chart(result.mentions), go.Figure)
    assert isinstance(build_positive_negative_chart(result.summary), go.Figure)
    assert isinstance(build_aspect_rating_chart(result.summary), go.Figure)


def test_rating_chart_hides_when_rating_is_unavailable(
    aspect_topic_df: pd.DataFrame,
) -> None:
    result = analyze_aspects(aspect_topic_df.drop(columns="rating"))
    assert build_aspect_rating_chart(result.summary) is None


def test_sentiment_colors_are_consistent_between_aspect_charts(
    aspect_topic_df: pd.DataFrame,
) -> None:
    result = analyze_aspects(aspect_topic_df)
    comparison = build_aspect_sentiment_chart(result.mentions)
    positive_negative = build_positive_negative_chart(result.summary)

    assert comparison is not None
    assert positive_negative is not None

    comparison_colors = {trace.name: trace.marker.color for trace in comparison.data}
    positive_negative_colors = {
        trace.name: trace.marker.color for trace in positive_negative.data
    }

    assert comparison_colors["Positive"] == positive_negative_colors["Positive"]
    assert comparison_colors["Negative"] == positive_negative_colors["Negative"]
    assert comparison_colors["Positive"] != comparison_colors["Negative"]
