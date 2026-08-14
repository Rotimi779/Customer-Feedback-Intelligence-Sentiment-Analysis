"""Tests for TF-IDF + Logistic Regression training, persistence, and confidence."""

from pathlib import Path

import pandas as pd

from src.sentiment.baseline import BaselineSentimentModel
from src.sentiment.train_baseline import train_baseline


def test_baseline_predicts_labels_and_probabilities(labeled_sentiment_df: pd.DataFrame) -> None:
    model = BaselineSentimentModel().fit(
        labeled_sentiment_df["review_text"],
        labeled_sentiment_df["sentiment_label"],
    )
    labels, scores = model.predict_with_confidence(
        ["I love this reliable product", "It works as expected", "I hate this terrible product"]
    )
    assert len(labels) == 3
    assert all(label in {"Negative", "Neutral", "Positive"} for label in labels)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_baseline_round_trip_artifacts(
    tmp_path: Path,
    labeled_sentiment_df: pd.DataFrame,
) -> None:
    artifact_dir = tmp_path / "baseline"
    model = BaselineSentimentModel().fit(
        labeled_sentiment_df["review_text"],
        labeled_sentiment_df["sentiment_label"],
    )
    expected = model.predict(["excellent and reliable", "terrible and unreliable"])
    model.save(artifact_dir)
    loaded = BaselineSentimentModel.load(artifact_dir)
    assert loaded.predict(["excellent and reliable", "terrible and unreliable"]) == expected


def test_train_baseline_writes_required_evaluation_artifacts(
    tmp_path: Path,
    labeled_sentiment_df: pd.DataFrame,
) -> None:
    artifact_dir = tmp_path / "trained"
    report = train_baseline(labeled_sentiment_df, artifact_dir=artifact_dir, dataset_name="unit-test")
    assert 0.0 <= report["macro_f1"] <= 1.0
    assert (artifact_dir / "tfidf_vectorizer.joblib").exists()
    assert (artifact_dir / "logistic_regression.joblib").exists()
    assert (artifact_dir / "metadata.json").exists()
    assert (artifact_dir / "metrics.json").exists()
    assert (artifact_dir / "test_predictions.json").exists()
