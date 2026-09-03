"""Integration tests from Phase 5 topic outputs into aspect analysis."""

import pandas as pd
import pytest

from src.pipeline import run_aspect_stage


def test_run_aspect_stage_preserves_upstream_outputs_and_appends_aspects(
    aspect_topic_df: pd.DataFrame,
) -> None:
    original = aspect_topic_df.copy(deep=True)
    result = run_aspect_stage(aspect_topic_df)

    pd.testing.assert_frame_equal(aspect_topic_df, original)
    assert len(result.dataframe) == len(original)
    assert {"topic_id", "topic_label", "detected_aspects", "aspect_sentiment"}.issubset(
        result.dataframe.columns
    )
    assert result.dataframe.loc[4, "detected_aspects"] == []


def test_run_aspect_stage_requires_phase5_topic_contract(
    aspect_topic_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="Phase 5 topic-enriched"):
        run_aspect_stage(aspect_topic_df.drop(columns="topic_label"))
