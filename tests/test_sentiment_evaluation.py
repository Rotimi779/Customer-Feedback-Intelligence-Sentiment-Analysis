"""Tests for model comparison and explicit production selection."""

from pathlib import Path

import pandas as pd

import src.sentiment.evaluation as evaluation


def test_model_comparison_uses_same_metric_rows() -> None:
    baseline = {
        "accuracy": 0.8,
        "precision_macro": 0.79,
        "recall_macro": 0.78,
        "macro_f1": 0.77,
        "weighted_f1": 0.80,
        "inference_seconds": 0.1,
        "samples_per_second": 100.0,
        "training_seconds": 1.0,
        "model_size_bytes": 1234,
    }
    transformer = {key: value for key, value in baseline.items()}
    comparison = evaluation.build_model_comparison(baseline, transformer)
    assert not comparison.empty
    assert list(comparison.columns) == ["Metric", "Logistic Regression", "DistilBERT"]
    assert "Macro F1" in comparison["Metric"].tolist()


def test_production_selection_requires_explicit_rationale(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "production_model.json"
    monkeypatch.setattr(evaluation, "PRODUCTION_SELECTION_PATH", path)
    evaluation.save_production_model_selection(
        "logistic_regression",
        "Comparable Macro F1 with lower inference latency.",
    )
    loaded = evaluation.load_production_model_selection()
    assert loaded == {
        "model_name": "logistic_regression",
        "rationale": "Comparable Macro F1 with lower inference latency.",
    }
