"""Reusable DistilBERT sentiment inference wrapper."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch

from src.sentiment.preprocessing import prepare_transformer_texts
from src.sentiment.utils import ID_TO_LABEL, SENTIMENT_LABELS

ProgressCallback = Callable[[int, int], None]


class TransformerDependencyError(RuntimeError):
    """Raised when Hugging Face Transformers is unavailable."""


class TransformerModelNotReadyError(RuntimeError):
    """Raised when local DistilBERT artifacts are unavailable or invalid."""


class DistilBertSentimentModel:
    """Three-class DistilBERT classifier with batched confidence outputs."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        device: str | torch.device | None = None,
        max_length: int = 256,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = max_length
        self.model.to(self.device)
        self.model.eval()
        self._validate_label_contract()

    @classmethod
    def load(
        cls,
        artifact_dir: Path,
        *,
        device: str | torch.device | None = None,
        max_length: int = 256,
    ) -> "DistilBertSentimentModel":
        """Load a fine-tuned local DistilBERT model and tokenizer."""
        if not artifact_dir.exists() or not (artifact_dir / "config.json").exists():
            raise TransformerModelNotReadyError(
                "The DistilBERT model has not been fine-tuned yet. "
                "Run src.sentiment.train_transformer first."
            )
        try:
            from transformers import AutoTokenizer, DistilBertForSequenceClassification
        except ImportError as exc:
            raise TransformerDependencyError(
                "Transformers is required for DistilBERT inference. "
                "Install the project requirements first."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(str(artifact_dir), local_files_only=True)
        model = DistilBertForSequenceClassification.from_pretrained(
            str(artifact_dir),
            local_files_only=True,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=max_length,
        )

    def predict_with_confidence(
        self,
        texts: Iterable[object],
        *,
        batch_size: int = 16,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[list[str], list[float]]:
        """Predict labels and max-softmax confidence in input order."""
        prepared = prepare_transformer_texts(texts)
        total = len(prepared)
        if total == 0:
            return [], []
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        labels: list[str] = []
        confidences: list[float] = []
        with torch.no_grad():
            for start in range(0, total, batch_size):
                batch = prepared[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(self.device)
                    for key, value in encoded.items()
                }
                logits = self.model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1)
                scores, indices = torch.max(probabilities, dim=-1)
                labels.extend(self._label_for_id(int(index)) for index in indices.cpu())
                confidences.extend(float(score) for score in scores.cpu())
                if progress_callback is not None:
                    progress_callback(min(start + len(batch), total), total)
        return labels, confidences

    def _label_for_id(self, label_id: int) -> str:
        configured = getattr(self.model.config, "id2label", {}) or {}
        raw = configured.get(label_id, configured.get(str(label_id), ID_TO_LABEL.get(label_id)))
        if raw is None:
            raise TransformerModelNotReadyError(f"No label mapping exists for class id {label_id}.")
        normalized = str(raw).strip().title()
        if normalized not in SENTIMENT_LABELS:
            raise TransformerModelNotReadyError(
                "DistilBERT artifacts do not use the required Negative/Neutral/Positive label contract."
            )
        return normalized

    def _validate_label_contract(self) -> None:
        configured = getattr(self.model.config, "id2label", {}) or {}
        if not configured:
            raise TransformerModelNotReadyError("DistilBERT config is missing id2label metadata.")
        normalized = {
            str(value).strip().title()
            for value in configured.values()
        }
        if normalized != set(SENTIMENT_LABELS):
            raise TransformerModelNotReadyError(
                "DistilBERT must be fine-tuned for exactly Negative, Neutral, and Positive."
            )
