from __future__ import annotations

import pandas as pd

from src.aspects.aggregation import analyze_aspects
from src.insights.generator import generate_business_insights


def test_executive_summary_uses_measured_outputs(insight_topic_df: pd.DataFrame) -> None:
    aspect_result = analyze_aspects(insight_topic_df)
    result = generate_business_insights(aspect_result.dataframe)

    assert "12 reviews" in result.executive_summary
    assert "Billing / Payment" in result.executive_summary
    assert "Billing" in result.executive_summary


def test_summary_remains_available_without_date_metadata(aspect_topic_df: pd.DataFrame) -> None:
    aspect_result = analyze_aspects(aspect_topic_df)
    result = generate_business_insights(aspect_result.dataframe)

    assert result.executive_summary
    assert result.trends.available is False
    assert "date" in str(result.trends.reason).lower()
