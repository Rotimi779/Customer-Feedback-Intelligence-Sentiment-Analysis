from __future__ import annotations

import pandas as pd

from src.aspects.aggregation import analyze_aspects
from src.insights.generator import generate_business_insights


def test_generator_produces_required_business_metrics(insight_topic_df: pd.DataFrame) -> None:
    aspect_result = analyze_aspects(insight_topic_df)
    result = generate_business_insights(
        aspect_result.dataframe,
        aspect_summary=aspect_result.summary,
        aspect_mentions=aspect_result.mentions,
    )

    assert result.metrics["total_reviews"] == 12
    assert result.metrics["most_discussed_topic"]["topic_label"] == "Billing / Payment"
    assert result.metrics["priority_improvement"]["aspect"] == "Billing"
    assert result.metrics["key_strength"] is not None
    assert not result.findings.empty
    assert not result.recommendations.empty


def test_generator_is_non_mutating(insight_topic_df: pd.DataFrame) -> None:
    aspect_result = analyze_aspects(insight_topic_df)
    source = aspect_result.dataframe.copy(deep=True)
    generate_business_insights(aspect_result.dataframe)
    pd.testing.assert_frame_equal(aspect_result.dataframe, source)


def test_every_evidence_review_id_exists_in_source(insight_topic_df: pd.DataFrame) -> None:
    aspect_result = analyze_aspects(insight_topic_df)
    result = generate_business_insights(aspect_result.dataframe)
    valid = set(aspect_result.dataframe["review_id"].astype(str))

    for ids in result.findings["representative_review_ids"]:
        assert set(ids).issubset(valid)
