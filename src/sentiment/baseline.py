"""Reusable TF-IDF + Logistic Regression sentiment model."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.sentiment.preprocessing import prepare_classical_texts
from src.sentiment.utils import (
    BaselineTrainingConfig,
    SENTIMENT_LABELS,
    read_json,
    write_json,
)

VECTORIZER_FILENAME = "tfidf_vectorizer.joblib"
MODEL_FILENAME = "logistic_regression.joblib"
METADATA_FILENAME = "metadata.json"


class BaselineModelNotReadyError(RuntimeError):
    """Raised when baseline artifacts are unavailable or incomplete."""


class BaselineSentimentModel:
    """TF-IDF + Logistic Regression model with a stable prediction interface."""

    def __init__(
        self,
        *,
        vectorizer: TfidfVectorizer | None = None,
        classifier: LogisticRegression | None = None,
        config: BaselineTrainingConfig = BaselineTrainingConfig(),
    ) -> None:
        self.config = config
        self.vectorizer = vectorizer or TfidfVectorizer(
            ngram_range=(config.ngram_min, config.ngram_max),
            max_features=config.max_features,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.classifier = classifier or LogisticRegression(
            C=config.regularization_c,
            max_iter=config.max_iterations,
            class_weight=config.class_weight,
            random_state=config.random_state,
        )

    @property
    def is_fitted(self) -> bool:
        """Return whether both scikit-learn components contain fitted state."""
        return hasattr(self.vectorizer, "vocabulary_") and hasattr(self.classifier, "classes_")

    def fit(self, texts: Iterable[object], labels: Iterable[str]) -> "BaselineSentimentModel":
        """Fit TF-IDF on training text only, then fit Logistic Regression."""
        prepared_texts = prepare_classical_texts(texts)
        label_values = list(labels)
        if len(prepared_texts) != len(label_values):
            raise ValueError("texts and labels must contain the same number of rows.")
        matrix = self.vectorizer.fit_transform(prepared_texts)
        self.classifier.fit(matrix, label_values)
        return self

    def predict(self, texts: Iterable[object]) -> list[str]:
        """Predict canonical sentiment labels in input order."""
        self._require_fitted()
        matrix = self.vectorizer.transform(prepare_classical_texts(texts))
        return [str(value) for value in self.classifier.predict(matrix).tolist()]

    def predict_with_confidence(self, texts: Iterable[object]) -> tuple[list[str], list[float]]:
        """Predict labels and maximum class probability for each text."""
        self._require_fitted()
        prepared = prepare_classical_texts(texts)
        matrix = self.vectorizer.transform(prepared)
        probabilities = self.classifier.predict_proba(matrix)
        best_indices = np.argmax(probabilities, axis=1)
        classes = np.asarray(self.classifier.classes_)
        labels = [str(classes[index]) for index in best_indices]
        scores = probabilities[np.arange(len(best_indices)), best_indices].astype(float).tolist()
        return labels, scores

    def save(self, artifact_dir: Path, *, metadata: dict[str, object] | None = None) -> None:
        """Persist vectorizer, classifier, and model metadata."""
        self._require_fitted()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, artifact_dir / VECTORIZER_FILENAME)
        joblib.dump(self.classifier, artifact_dir / MODEL_FILENAME)
        payload: dict[str, object] = {
            "model_name": "tfidf_logistic_regression",
            "labels": list(SENTIMENT_LABELS),
            "training_config": asdict(self.config),
        }
        if metadata:
            payload.update(metadata)
        write_json(artifact_dir / METADATA_FILENAME, payload)

    @classmethod
    def load(cls, artifact_dir: Path) -> "BaselineSentimentModel":
        """Load saved baseline artifacts from a local directory."""
        vectorizer_path = artifact_dir / VECTORIZER_FILENAME
        model_path = artifact_dir / MODEL_FILENAME
        if not vectorizer_path.exists() or not model_path.exists():
            raise BaselineModelNotReadyError(
                "The Logistic Regression model has not been trained yet. "
                "Run src.sentiment.train_baseline first."
            )

        vectorizer = joblib.load(vectorizer_path)
        classifier = joblib.load(model_path)
        config = BaselineTrainingConfig()
        metadata_path = artifact_dir / METADATA_FILENAME
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            raw_config = metadata.get("training_config")
            if isinstance(raw_config, dict):
                config = BaselineTrainingConfig(**raw_config)
        return cls(vectorizer=vectorizer, classifier=classifier, config=config)

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise BaselineModelNotReadyError("The Logistic Regression baseline is not fitted.")
