"""Tests for aspect mention expansion and aggregate business metrics."""

import pandas as pd

from src.aspects.aggregation import analyze_aspects


def test_analysis_builds_frequency_sentiment_and_rating_metrics(
    aspect_topic_df: pd.DataFrame,
) -> None:
    result = analyze_aspects(aspect_topic_df)

    assert not result.mentions.empty
    assert not result.summary.empty
    battery = result.summary.loc[result.summary["aspect"].eq("Battery")].iloc[0]
    assert battery["mention_count"] == 1
    assert battery["positive_share"] == 1.0
    assert battery["average_rating"] == 5.0


def test_analysis_reports_coverage_and_multi_aspect_reviews(
    aspect_topic_df: pd.DataFrame,
) -> None:
    result = analyze_aspects(aspect_topic_df)

    assert result.evaluation["reviews_with_aspects"] == 4
    assert result.evaluation["aspect_coverage"] == 0.8
    assert result.evaluation["multi_aspect_reviews"] >= 3
    assert result.evaluation["sentiment_association_complete"] is True
    assert result.evaluation["manual_review_required"] is True
