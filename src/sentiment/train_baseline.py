"""Train and evaluate the TF-IDF + Logistic Regression sentiment baseline."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.sentiment.baseline import BaselineSentimentModel
from src.sentiment.evaluation import save_evaluation_report
from src.sentiment.metrics import compute_classification_metrics
from src.sentiment.utils import (
    DEFAULT_BASELINE_DIR,
    BaselineTrainingConfig,
    SentimentSplitConfig,
    dataframe_fingerprint,
    dataclass_to_dict,
    directory_size_bytes,
    parse_label_mapping_json,
    prepare_labeled_dataframe,
    set_global_seed,
    split_labeled_dataframe,
    write_json,
)


def train_baseline(
    dataframe: pd.DataFrame,
    *,
    text_column: str = "review_text",
    label_column: str = "sentiment_label",
    label_mapping: dict[str, str] | None = None,
    artifact_dir: Path = DEFAULT_BASELINE_DIR,
    training_config: BaselineTrainingConfig = BaselineTrainingConfig(),
    split_config: SentimentSplitConfig = SentimentSplitConfig(),
    dataset_name: str = "labelled_sentiment_dataset",
) -> dict[str, Any]:
    """Train, evaluate, and persist the classical sentiment baseline."""
    prepared = prepare_labeled_dataframe(
        dataframe,
        text_column=text_column,
        label_column=label_column,
        label_mapping=label_mapping,
    )
    train_df, validation_df, test_df = split_labeled_dataframe(
        prepared,
        text_column=text_column,
        label_column=label_column,
        config=split_config,
    )
    set_global_seed(training_config.random_state)

    model = BaselineSentimentModel(config=training_config)
    training_start = time.perf_counter()
    model.fit(train_df[text_column], train_df[label_column])
    training_seconds = time.perf_counter() - training_start

    validation_labels, _ = model.predict_with_confidence(validation_df[text_column])
    validation_metrics = compute_classification_metrics(
        validation_df[label_column].tolist(),
        validation_labels,
    )

    inference_start = time.perf_counter()
    test_labels, test_scores = model.predict_with_confidence(test_df[text_column])
    inference_seconds = time.perf_counter() - inference_start
    test_metrics = compute_classification_metrics(test_df[label_column].tolist(), test_labels)

    model_metadata = {
        "dataset_name": dataset_name,
        "dataset_fingerprint": dataframe_fingerprint(
            prepared,
            text_column=text_column,
            label_column=label_column,
        ),
        "split_config": dataclass_to_dict(split_config),
        "train_rows": len(train_df),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
        "class_distribution": prepared[label_column].value_counts().to_dict(),
        "validation_macro_f1": validation_metrics["macro_f1"],
    }
    model.save(artifact_dir, metadata=model_metadata)

    model_size_bytes = directory_size_bytes(artifact_dir)
    samples_per_second = 0.0 if inference_seconds <= 0 else len(test_df) / inference_seconds
    report: dict[str, Any] = {
        "model_name": "logistic_regression",
        "dataset_name": dataset_name,
        **test_metrics,
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "samples_per_second": float(samples_per_second),
        "model_size_bytes": int(model_size_bytes),
        "mean_confidence": float(sum(test_scores) / len(test_scores)),
        "validation_metrics": validation_metrics,
        "training_config": dataclass_to_dict(training_config),
        "split_config": dataclass_to_dict(split_config),
        "test_rows": len(test_df),
    }
    save_evaluation_report(report, artifact_dir / "metrics.json")
    write_json(
        artifact_dir / "test_predictions.json",
        {
            "review_text": test_df[text_column].tolist(),
            "actual": test_df[label_column].tolist(),
            "predicted": test_labels,
            "confidence": test_scores,
        },
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression sentiment baseline.")
    parser.add_argument("--input", required=True, type=Path, help="Labelled CSV path.")
    parser.add_argument("--text-column", default="review_text")
    parser.add_argument("--label-column", default="sentiment_label")
    parser.add_argument(
        "--label-map-json",
        help='Optional explicit JSON mapping, e.g. \'{"0":"Negative","1":"Neutral","2":"Positive"}\'.',
    )
    parser.add_argument("--dataset-name", default="labelled_sentiment_dataset")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    return parser


def main() -> None:
    """CLI entry point."""
    args = _build_parser().parse_args()
    dataframe = pd.read_csv(args.input)
    report = train_baseline(
        dataframe,
        text_column=args.text_column,
        label_column=args.label_column,
        label_mapping=parse_label_mapping_json(args.label_map_json),
        artifact_dir=args.artifact_dir,
        dataset_name=args.dataset_name,
    )
    print(
        "Baseline complete | "
        f"accuracy={report['accuracy']:.4f} | macro_f1={report['macro_f1']:.4f} | "
        f"inference_seconds={report['inference_seconds']:.4f}"
    )


if __name__ == "__main__":
    main()
