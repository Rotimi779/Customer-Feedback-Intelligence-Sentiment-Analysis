"""Single sentiment prediction interface used by the rest of the application."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from src.sentiment.baseline import BaselineSentimentModel
from src.sentiment.transformer import DistilBertSentimentModel
from src.sentiment.utils import DEFAULT_BASELINE_DIR, DEFAULT_DISTILBERT_DIR

ProgressCallback = Callable[[int, int], None]


class SentimentModelName(str, Enum):
    """Supported MVP sentiment models."""

    LOGISTIC_REGRESSION = "logistic_regression"
    DISTILBERT = "distilbert"

    @property
    def display_name(self) -> str:
        if self is SentimentModelName.LOGISTIC_REGRESSION:
            return "TF-IDF + Logistic Regression"
        return "DistilBERT"


@dataclass(frozen=True)
class SentimentInferenceResult:
    """Enriched DataFrame plus inference metadata."""

    dataframe: pd.DataFrame
    model_name: SentimentModelName
    mean_confidence: float


class SentimentAnalyzer:
    """Stable facade around either local sentiment model implementation."""

    def __init__(
        self,
        model_name: SentimentModelName,
        model: Any,
        *,
        transformer_batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.model = model
        self.transformer_batch_size = transformer_batch_size

    @classmethod
    def load(
        cls,
        model_name: SentimentModelName | str,
        *,
        baseline_dir: Path = DEFAULT_BASELINE_DIR,
        transformer_dir: Path = DEFAULT_DISTILBERT_DIR,
        transformer_batch_size: int = 16,
        device: str | None = None,
    ) -> "SentimentAnalyzer":
        """Load one local model from the standard artifact directories."""
        selected = SentimentModelName(model_name)
        if selected is SentimentModelName.LOGISTIC_REGRESSION:
            model = BaselineSentimentModel.load(baseline_dir)
        else:
            model = DistilBertSentimentModel.load(transformer_dir, device=device)
        return cls(
            selected,
            model,
            transformer_batch_size=transformer_batch_size,
        )

    def predict_texts(
        self,
        texts: list[object],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[list[str], list[float]]:
        """Predict standardized labels and confidence scores."""
        if self.model_name is SentimentModelName.LOGISTIC_REGRESSION:
            labels, scores = self.model.predict_with_confidence(texts)
            if progress_callback is not None:
                progress_callback(len(texts), len(texts))
            return labels, scores
        return self.model.predict_with_confidence(
            texts,
            batch_size=self.transformer_batch_size,
            progress_callback=progress_callback,
        )

    def predict_dataframe(
        self,
        dataframe: pd.DataFrame,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> SentimentInferenceResult:
        """Return a copy enriched with ``sentiment_label`` and ``sentiment_score``.

        Logistic Regression consumes ``clean_text``; DistilBERT consumes the
        minimally changed ``review_text`` column, matching the technical design.
        """
        required = {"review_text", "clean_text"}
        missing = sorted(required - set(dataframe.columns))
        if missing:
            raise ValueError(f"Canonical DataFrame is missing required columns: {missing}.")
        if dataframe.empty:
            raise ValueError("Sentiment inference requires at least one review.")

        text_column = (
            "clean_text"
            if self.model_name is SentimentModelName.LOGISTIC_REGRESSION
            else "review_text"
        )
        texts = dataframe[text_column].tolist()
        labels, scores = self.predict_texts(
            texts,
            progress_callback=progress_callback,
        )
        if len(labels) != len(dataframe) or len(scores) != len(dataframe):
            raise RuntimeError("Sentiment model returned an unexpected number of predictions.")

        enriched = dataframe.copy()
        enriched["sentiment_label"] = labels
        enriched["sentiment_score"] = pd.Series(scores, index=enriched.index, dtype="float64")
        return SentimentInferenceResult(
            dataframe=enriched,
            model_name=self.model_name,
            mean_confidence=float(enriched["sentiment_score"].mean()),
        )


def baseline_model_available(path: Path = DEFAULT_BASELINE_DIR) -> bool:
    """Return whether the required local baseline artifacts exist."""
    return (
        (path / "tfidf_vectorizer.joblib").exists()
        and (path / "logistic_regression.joblib").exists()
    )


def transformer_model_available(path: Path = DEFAULT_DISTILBERT_DIR) -> bool:
    """Return whether a local Hugging Face DistilBERT export exists."""
    return (path / "config.json").exists() and any(
        candidate.exists()
        for candidate in (path / "model.safetensors", path / "pytorch_model.bin")
    )


def available_sentiment_models() -> list[SentimentModelName]:
    """Return locally trained models in stable display order."""
    available: list[SentimentModelName] = []
    if baseline_model_available():
        available.append(SentimentModelName.LOGISTIC_REGRESSION)
    if transformer_model_available():
        available.append(SentimentModelName.DISTILBERT)
    return available
