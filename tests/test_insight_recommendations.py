from __future__ import annotations

import pandas as pd

from src.aspects.aggregation import analyze_aspects
from src.insights.generator import generate_business_insights


def test_recommendations_are_evidence_backed(insight_topic_df: pd.DataFrame) -> None:
    aspect_result = analyze_aspects(insight_topic_df)
    result = generate_business_insights(aspect_result.dataframe)

    required = {
        "priority",
        "title",
        "affected_item",
        "supporting_metric",
        "supporting_value",
        "supporting_count",
        "explanation",
    }
    assert required.issubset(result.recommendations.columns)
    assert result.recommendations["supporting_count"].gt(0).all()
    assert result.recommendations["explanation"].str.len().gt(20).all()


def test_billing_is_high_priority_when_negative_support_is_strong(
    insight_topic_df: pd.DataFrame,
) -> None:
    aspect_result = analyze_aspects(insight_topic_df)
    result = generate_business_insights(aspect_result.dataframe)
    billing = result.recommendations.loc[result.recommendations["affected_item"].eq("Billing")]

    assert not billing.empty
    assert "High" in set(billing["priority"])
