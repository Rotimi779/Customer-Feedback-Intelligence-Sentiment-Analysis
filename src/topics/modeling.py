"""NMF topic-model training, assignment, persistence, and CLI support."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from src.topics.labeling import extract_topic_keywords, generate_topic_labels
from src.topics.preprocessing import TopicCorpus, clean_topic_text, prepare_topic_corpus
from src.topics.utils import TopicModelConfig, TopicModelError, build_topic_summary

LOGGER = logging.getLogger(__name__)


@dataclass
class TopicModelResult:
    """Outputs from fitting and applying one NMF topic model."""

    dataframe: pd.DataFrame
    summary: pd.DataFrame
    topic_keywords: dict[int, list[str]]
    topic_labels: dict[int, str]
    representative_review_ids: dict[int, list[str]]
    assignment_strengths: np.ndarray
    model: "NMFTopicModel"


class NMFTopicModel:
    """Dataset-specific TF-IDF + NMF topic model for the MVP."""

    def __init__(self, config: TopicModelConfig | None = None) -> None:
        self.config = config or TopicModelConfig()
        self.vectorizer: TfidfVectorizer | None = None
        self.model: NMF | None = None
        self.topic_keywords: dict[int, list[str]] = {}
        self.topic_labels: dict[int, str] = {}
        self.training_metadata: dict[str, Any] = {}

    @property
    def is_fitted(self) -> bool:
        return self.vectorizer is not None and self.model is not None

    def _build_vectorizer(self, document_count: int) -> TfidfVectorizer:
        min_df = self.config.min_df if document_count >= 20 else 1
        max_df = self.config.max_df
        if isinstance(max_df, float) and int(max_df * document_count) < min_df:
            max_df = 1.0

        return TfidfVectorizer(
            preprocessor=clean_topic_text,
            lowercase=False,
            stop_words="english",
            ngram_range=self.config.ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=self.config.max_features,
            sublinear_tf=True,
        )

    def fit(self, texts: list[str]) -> "NMFTopicModel":
        """Fit TF-IDF and NMF on prepared unique review text."""
        if len(texts) < self.config.n_topics:
            raise TopicModelError(
                f"Need at least {self.config.n_topics} usable unique reviews to fit "
                f"{self.config.n_topics} topics; received {len(texts)}."
            )

        vectorizer = self._build_vectorizer(len(texts))
        matrix = vectorizer.fit_transform(texts)
        n_features = matrix.shape[1]
        if n_features < self.config.n_topics:
            raise TopicModelError(
                f"Only {n_features} usable TF-IDF features remain, which is fewer "
                f"than the requested {self.config.n_topics} topics."
            )

        model = NMF(
            n_components=self.config.n_topics,
            init="nndsvda",
            random_state=self.config.random_state,
            max_iter=self.config.max_iter,
        )
        model.fit(matrix)

        feature_names = vectorizer.get_feature_names_out()
        keywords = extract_topic_keywords(
            model.components_,
            feature_names,
            top_n=self.config.top_n_words,
        )

        self.vectorizer = vectorizer
        self.model = model
        self.topic_keywords = keywords
        self.topic_labels = generate_topic_labels(keywords)
        self.training_metadata = {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "document_count": len(texts),
            "feature_count": int(n_features),
            "reconstruction_error": float(model.reconstruction_err_),
            "config": self.config.as_dict(),
        }
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        """Return document-topic weights for unseen reviews."""
        if not self.is_fitted or self.vectorizer is None or self.model is None:
            raise TopicModelError("Topic model must be fitted before transform().")
        matrix = self.vectorizer.transform(texts)
        zero_rows = np.asarray(matrix.getnnz(axis=1)).ravel() == 0
        if bool(np.any(zero_rows)):
            raise TopicModelError(
                "At least one review contains no vocabulary recognized by the fitted "
                "topic model. Adjust preprocessing/configuration before assignment."
            )
        return np.asarray(self.model.transform(matrix), dtype=float)

    def fit_dataframe(
        self,
        dataframe: pd.DataFrame,
        *,
        text_column: str | None = None,
        representative_count: int = 5,
    ) -> TopicModelResult:
        """Fit the model and append topic_id/topic_label to a DataFrame copy."""
        corpus: TopicCorpus = prepare_topic_corpus(dataframe, text_column=text_column)
        self.fit(corpus.texts)
        weights = self.transform(corpus.texts)
        assignments = weights.argmax(axis=1).astype(int)
        strengths = weights.max(axis=1)

        enriched = corpus.dataframe.copy()
        enriched["topic_id"] = assignments
        enriched["topic_label"] = [self.topic_labels[int(item)] for item in assignments]

        summary = build_topic_summary(
            enriched,
            topic_keywords=self.topic_keywords,
            topic_labels=self.topic_labels,
        )
        representatives = self._representative_review_ids(
            enriched,
            assignments,
            strengths,
            count=representative_count,
        )

        self.training_metadata.update(
            {
                "source_rows": int(len(dataframe)),
                "modeled_rows": int(len(enriched)),
                "empty_rows_removed": corpus.empty_rows_removed,
                "duplicate_rows_removed": corpus.duplicate_rows_removed,
            }
        )

        return TopicModelResult(
            dataframe=enriched,
            summary=summary,
            topic_keywords=self.topic_keywords.copy(),
            topic_labels=self.topic_labels.copy(),
            representative_review_ids=representatives,
            assignment_strengths=strengths,
            model=self,
        )

    @staticmethod
    def _representative_review_ids(
        dataframe: pd.DataFrame,
        assignments: np.ndarray,
        strengths: np.ndarray,
        *,
        count: int,
    ) -> dict[int, list[str]]:
        representatives: dict[int, list[str]] = {}
        id_values = (
            dataframe["review_id"].astype(str).tolist()
            if "review_id" in dataframe.columns
            else [str(index) for index in dataframe.index]
        )
        for topic_id in sorted(set(assignments.tolist())):
            candidate_indices = np.where(assignments == topic_id)[0]
            ranked = candidate_indices[np.argsort(strengths[candidate_indices])[::-1]]
            representatives[int(topic_id)] = [id_values[index] for index in ranked[:count]]
        return representatives

    def save(
        self,
        directory: str | Path,
        *,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Persist the vectorizer, NMF model, and transparent metadata locally."""
        if not self.is_fitted or self.vectorizer is None or self.model is None:
            raise TopicModelError("Fit the topic model before saving artifacts.")

        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, output / "tfidf_vectorizer.joblib")
        joblib.dump(self.model, output / "nmf_model.joblib")

        metadata = {
            **self.training_metadata,
            "topic_keywords": {str(k): v for k, v in self.topic_keywords.items()},
            "topic_labels": {str(k): v for k, v in self.topic_labels.items()},
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        (output / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return output

    @classmethod
    def load(cls, directory: str | Path) -> "NMFTopicModel":
        """Load a locally saved topic model."""
        source = Path(directory)
        metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
        config_payload = dict(metadata.get("config", {}))
        if "ngram_range" in config_payload:
            config_payload["ngram_range"] = tuple(config_payload["ngram_range"])
        instance = cls(TopicModelConfig(**config_payload))
        instance.vectorizer = joblib.load(source / "tfidf_vectorizer.joblib")
        instance.model = joblib.load(source / "nmf_model.joblib")
        instance.topic_keywords = {
            int(key): list(value) for key, value in metadata["topic_keywords"].items()
        }
        instance.topic_labels = {
            int(key): str(value) for key, value in metadata["topic_labels"].items()
        }
        instance.training_metadata = metadata
        return instance


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the MVP NMF topic model.")
    parser.add_argument("--input", required=True, help="CSV containing review_text or clean_text.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--n-topics", type=int, default=8)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--stability-runs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Train/save a topic model on a public experiment dataset."""
    from src.topics.evaluation import evaluate_topic_model, save_topic_evaluation

    args = _parse_args()
    dataframe = pd.read_csv(args.input)
    config = replace(TopicModelConfig(), n_topics=args.n_topics)
    model = NMFTopicModel(config)
    result = model.fit_dataframe(dataframe)

    output_dir = Path(args.output_dir or f"models/topic_model/{args.dataset_name}")
    metrics = evaluate_topic_model(
        model,
        result.dataframe,
        stability_runs=args.stability_runs,
    )
    model.save(
        output_dir,
        extra_metadata={"dataset_name": args.dataset_name, "evaluation": metrics},
    )
    result.dataframe.to_csv(output_dir / "topic_assignments.csv", index=False)
    result.summary.to_csv(output_dir / "topic_summary.csv", index=False)
    save_topic_evaluation(metrics, output_dir / "evaluation.json")

    print(result.summary.to_string(index=False))
    print("\nEvaluation")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"\nSaved topic artifacts to {output_dir}")


if __name__ == "__main__":
    main()
