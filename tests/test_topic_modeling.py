"""Tests for NMF training, assignment, and local persistence."""

from pathlib import Path

import pandas as pd

from src.topics.modeling import NMFTopicModel
from src.topics.utils import TopicModelConfig


def test_nmf_assigns_every_review_and_preserves_sentiment(topic_sentiment_df: pd.DataFrame) -> None:
    model = NMFTopicModel(TopicModelConfig(n_topics=3, min_df=1, top_n_words=5))
    result = model.fit_dataframe(topic_sentiment_df)

    assert len(result.dataframe) == len(topic_sentiment_df)
    assert {"topic_id", "topic_label"}.issubset(result.dataframe.columns)
    assert result.dataframe["topic_id"].between(0, 2).all()
    assert result.dataframe["topic_label"].notna().all()
    assert result.dataframe["sentiment_label"].equals(topic_sentiment_df["sentiment_label"])
    assert len(result.summary) == 3
    assert set(result.topic_keywords) == {0, 1, 2}


def test_nmf_round_trip_persistence(
    topic_sentiment_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    model = NMFTopicModel(TopicModelConfig(n_topics=3, min_df=1, top_n_words=5))
    result = model.fit_dataframe(topic_sentiment_df)
    artifact_dir = model.save(tmp_path / "topic_model", extra_metadata={"dataset_name": "test"})

    loaded = NMFTopicModel.load(artifact_dir)
    weights = loaded.transform(result.dataframe["clean_text"].astype(str).tolist())

    assert weights.shape == (len(result.dataframe), 3)
    assert loaded.topic_labels == model.topic_labels
    assert (artifact_dir / "tfidf_vectorizer.joblib").exists()
    assert (artifact_dir / "nmf_model.joblib").exists()
    assert (artifact_dir / "metadata.json").exists()
