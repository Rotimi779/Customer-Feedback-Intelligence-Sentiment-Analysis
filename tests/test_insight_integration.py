from __future__ import annotations

import pandas as pd
import pytest

from src.aspects.aggregation import analyze_aspects
from src.pipeline import run_insight_stage


def test_pipeline_insight_stage_accepts_phase6_results(insight_topic_df: pd.DataFrame) -> None:
    aspect_result = analyze_aspects(insight_topic_df)
    result = run_insight_stage(
        aspect_result.dataframe,
        aspect_summary=aspect_result.summary,
        aspect_mentions=aspect_result.mentions,
    )

    assert result.metrics["total_reviews"] == len(insight_topic_df)
    assert result.executive_summary
    assert not result.findings.empty


def test_pipeline_insight_stage_rejects_pre_aspect_data(insight_topic_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Phase 6"):
        run_insight_stage(insight_topic_df)
