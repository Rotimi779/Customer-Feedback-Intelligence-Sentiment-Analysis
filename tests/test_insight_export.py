from __future__ import annotations

import pandas as pd

from src.aspects.aggregation import analyze_aspects
from src.insights.export import build_markdown_report, dataframe_to_csv_bytes, recommendations_to_csv_bytes
from src.insights.generator import generate_business_insights


def test_exports_are_nonempty_and_include_final_fields(insight_topic_df: pd.DataFrame) -> None:
    aspect_result = analyze_aspects(insight_topic_df)
    result = generate_business_insights(aspect_result.dataframe)

    csv_bytes = dataframe_to_csv_bytes(result.dataframe)
    recommendations = recommendations_to_csv_bytes(result.recommendations)
    report = build_markdown_report(result)

    assert b"detected_aspects" in csv_bytes
    assert b"supporting_metric" in recommendations
    assert "# Customer Feedback Business Insights" in report
    assert "## Recommendations" in report
    assert "does not establish causal" in report or "do not establish causal" in report
