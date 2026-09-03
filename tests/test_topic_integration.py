"""Integration test from sentiment-enriched data into topic outputs."""

import pandas as pd

from src.pipeline import run_topic_stage
from src.topics.utils import TopicModelConfig


def test_run_topic_stage_appends_only_topic_columns(topic_sentiment_df: pd.DataFrame) -> None:
    original = topic_sentiment_df.copy(deep=True)
    result = run_topic_stage(
        topic_sentiment_df,
        config=TopicModelConfig(n_topics=3, min_df=1, top_n_words=5),
        stability_runs=1,
    )

    pd.testing.assert_frame_equal(topic_sentiment_df, original)
    assert {"topic_id", "topic_label"}.issubset(result.dataframe.columns)
    assert len(result.dataframe) == len(original)
    evaluation = result.model.training_metadata["evaluation"]
    assert evaluation["all_reviews_assigned"] is True
