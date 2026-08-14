"""Common evaluation metrics for both sentiment models."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.sentiment.utils import SENTIMENT_LABELS


def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str] = SENTIMENT_LABELS,
) -> dict[str, object]:
    """Compute the metrics required by EXPERIMENTS.md."""
    true_values = list(y_true)
    predicted_values = list(y_pred)
    ordered_labels = list(labels)
    if len(true_values) != len(predicted_values):
        raise ValueError("y_true and y_pred must have the same length.")
    if not true_values:
        raise ValueError("At least one prediction is required for evaluation.")

    return {
        "accuracy": float(accuracy_score(true_values, predicted_values)),
        "precision_macro": float(
            precision_score(
                true_values,
                predicted_values,
                labels=ordered_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                true_values,
                predicted_values,
                labels=ordered_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                true_values,
                predicted_values,
                labels=ordered_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                true_values,
                predicted_values,
                labels=ordered_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            true_values,
            predicted_values,
            labels=ordered_labels,
        ).tolist(),
        "labels": ordered_labels,
        "classification_report": classification_report(
            true_values,
            predicted_values,
            labels=ordered_labels,
            output_dict=True,
            zero_division=0,
        ),
    }
