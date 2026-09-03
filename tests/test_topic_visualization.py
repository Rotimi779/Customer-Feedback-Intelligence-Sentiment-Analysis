"""Tests for reusable Phase 5 Plotly builders."""

import pandas as pd

from src.topics.modeling import NMFTopicModel
from src.topics.utils import TopicModelConfig
from src.topics.visualization import (
    build_topic_distribution_chart,
    build_topic_frequency_chart,
    build_topic_keyword_table,
    build_topic_sentiment_chart,
)


def test_topic_visualizations_build_from_topic_results(topic_sentiment_df: pd.DataFrame) -> None:
    result = NMFTopicModel(
        TopicModelConfig(n_topics=3, min_df=1, top_n_words=5)
    ).fit_dataframe(topic_sentiment_df)

    frequency = build_topic_frequency_chart(result.summary)
    distribution = build_topic_distribution_chart(result.summary)
    sentiment = build_topic_sentiment_chart(result.dataframe)
    table = build_topic_keyword_table(result.summary)

    assert frequency.data
    assert distribution.data
    assert sentiment is not None and sentiment.data
    assert len(table) == 3
