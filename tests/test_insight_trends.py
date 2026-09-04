from __future__ import annotations

import pandas as pd

from src.aspects.aggregation import analyze_aspects
from src.insights.trends import analyze_trends


def test_trends_detect_growth_and_worsening_aspect(insight_topic_df: pd.DataFrame) -> None:
    aspect_result = analyze_aspects(insight_topic_df)
    trends = analyze_trends(
        aspect_result.dataframe,
        aspect_mentions=aspect_result.mentions,
    )

    assert trends.available is True
    assert trends.fastest_growing_topic is not None
    assert trends.fastest_growing_topic["topic_label"] == "Billing / Payment"
    assert trends.worsening_aspect is not None
    assert trends.worsening_aspect["aspect"] == "Billing"
    assert trends.worsening_aspect["negative_share_change"] > 0


def test_trends_return_clear_reason_without_dates(aspect_topic_df: pd.DataFrame) -> None:
    aspect_result = analyze_aspects(aspect_topic_df)
    trends = analyze_trends(aspect_result.dataframe, aspect_mentions=aspect_result.mentions)

    assert trends.available is False
    assert trends.reason
    assert trends.review_volume.empty


def test_sentiment_trend_shares_are_bounded(insight_topic_df: pd.DataFrame) -> None:
    aspect_result = analyze_aspects(insight_topic_df)
    trends = analyze_trends(aspect_result.dataframe, aspect_mentions=aspect_result.mentions)

    assert trends.sentiment["share"].between(0, 1).all()
