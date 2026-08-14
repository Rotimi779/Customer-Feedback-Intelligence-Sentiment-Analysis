"""Evaluation report persistence and model-comparison utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.sentiment.utils import (
    DEFAULT_BASELINE_DIR,
    DEFAULT_DISTILBERT_DIR,
    PRODUCTION_SELECTION_PATH,
    read_json,
    write_json,
)

METRIC_DISPLAY_ORDER: tuple[tuple[str, str], ...] = (
    ("accuracy", "Accuracy"),
    ("precision_macro", "Precision (macro)"),
    ("recall_macro", "Recall (macro)"),
    ("macro_f1", "Macro F1"),
    ("weighted_f1", "Weighted F1"),
    ("inference_seconds", "Inference time (s)"),
    ("samples_per_second", "Samples / second"),
    ("training_seconds", "Training time (s)"),
    ("model_size_bytes", "Model size (bytes)"),
)


def save_evaluation_report(report: dict[str, Any], path: Path) -> None:
    """Persist one model's evaluation report."""
    write_json(path, report)


def load_evaluation_report(path: Path) -> dict[str, Any] | None:
    """Return a saved report, or ``None`` when the model is not evaluated yet."""
    if not path.exists():
        return None
    return read_json(path)


def build_model_comparison(
    baseline_report: dict[str, Any] | None,
    transformer_report: dict[str, Any] | None,
) -> pd.DataFrame:
    """Build the common Logistic Regression vs DistilBERT comparison table."""
    if baseline_report is None and transformer_report is None:
        return pd.DataFrame(columns=["Metric", "Logistic Regression", "DistilBERT"])

    rows: list[dict[str, Any]] = []
    for key, display_name in METRIC_DISPLAY_ORDER:
        rows.append(
            {
                "Metric": display_name,
                "Logistic Regression": None if baseline_report is None else baseline_report.get(key),
                "DistilBERT": None if transformer_report is None else transformer_report.get(key),
            }
        )
    return pd.DataFrame(rows)


def load_saved_model_comparison(
    *,
    baseline_dir: Path = DEFAULT_BASELINE_DIR,
    transformer_dir: Path = DEFAULT_DISTILBERT_DIR,
) -> pd.DataFrame:
    """Load the two standard reports from the local model directories."""
    return build_model_comparison(
        load_evaluation_report(baseline_dir / "metrics.json"),
        load_evaluation_report(transformer_dir / "metrics.json"),
    )


def save_production_model_selection(model_name: str, rationale: str) -> None:
    """Record the explicit human-reviewed production-model decision."""
    valid_models = {"logistic_regression", "distilbert"}
    if model_name not in valid_models:
        raise ValueError(f"model_name must be one of {sorted(valid_models)}.")
    rationale = rationale.strip()
    if not rationale:
        raise ValueError("A short selection rationale is required.")
    write_json(
        PRODUCTION_SELECTION_PATH,
        {"model_name": model_name, "rationale": rationale},
    )


def load_production_model_selection() -> dict[str, Any] | None:
    """Load the saved production-model choice when it exists."""
    if not PRODUCTION_SELECTION_PATH.exists():
        return None
    return read_json(PRODUCTION_SELECTION_PATH)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect sentiment model metrics or record the production-model choice."
    )
    parser.add_argument(
        "--production-model",
        choices=("logistic_regression", "distilbert"),
        help="Record the reviewed production-model choice.",
    )
    parser.add_argument(
        "--rationale",
        help="Short evidence-based reason for the production-model choice.",
    )
    return parser


def main() -> None:
    """CLI for comparison inspection and explicit model selection."""
    args = _build_parser().parse_args()
    comparison = load_saved_model_comparison()
    if comparison.empty:
        print("No saved sentiment evaluation reports were found.")
    else:
        print(comparison.to_string(index=False))

    if args.production_model:
        if not args.rationale:
            raise SystemExit("--rationale is required with --production-model.")
        save_production_model_selection(args.production_model, args.rationale)
        print(f"Saved production model selection: {args.production_model}")


if __name__ == "__main__":
    main()
