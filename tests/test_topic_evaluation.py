"""Tests for topic-quality metrics required by the experiments plan."""

import pandas as pd

from src.topics.evaluation import evaluate_topic_model
from src.topics.modeling import NMFTopicModel
from src.topics.utils import TopicModelConfig


def test_topic_evaluation_reports_required_metrics(topic_sentiment_df: pd.DataFrame) -> None:
    model = NMFTopicModel(TopicModelConfig(n_topics=3, min_df=1, top_n_words=5))
    result = model.fit_dataframe(topic_sentiment_df)
    metrics = evaluate_topic_model(model, result.dataframe, stability_runs=2)

    assert metrics["number_of_topics"] == 3
    assert metrics["all_reviews_assigned"] is True
    assert 0.0 <= float(metrics["topic_coverage"]) <= 1.0
    assert -1.0 <= float(metrics["topic_coherence_npmi"]) <= 1.0
    assert 0.0 <= float(metrics["topic_diversity"]) <= 1.0
    assert 0.0 <= float(metrics["topic_stability"]) <= 1.0
    assert metrics["manual_interpretability_review_required"] is True
