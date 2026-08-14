"""Tests for the common evaluation contract."""

from src.sentiment.metrics import compute_classification_metrics


def test_perfect_predictions_have_perfect_required_metrics() -> None:
    labels = ["Negative", "Neutral", "Positive"] * 2
    metrics = compute_classification_metrics(labels, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision_macro"] == 1.0
    assert metrics["recall_macro"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
