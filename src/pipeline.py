"""Application pipeline contracts and completed stage orchestration."""

from __future__ import annotations

import pandas as pd

from src.ingestion.cleaning import build_canonical_dataframe
from src.ingestion.schema import (
    DEFAULT_INGESTION_CONFIG,
    CanonicalizationResult,
    ColumnMapping,
    IngestionConfig,
)
from src.ingestion.validator import DatasetValidationError, validate_dataset
from src.sentiment import SentimentAnalyzer, SentimentInferenceResult, SentimentModelName
from src.topics.evaluation import evaluate_topic_model
from src.topics.modeling import NMFTopicModel, TopicModelResult
from src.topics.utils import TopicModelConfig


def prepare_dataset(
    dataframe: pd.DataFrame,
    mapping: ColumnMapping,
    *,
    config: IngestionConfig = DEFAULT_INGESTION_CONFIG,
) -> CanonicalizationResult:
    """Validate source data and create the canonical ingestion DataFrame."""
    report = validate_dataset(dataframe, mapping, config=config)
    if not report.is_valid:
        raise DatasetValidationError(report)
    return build_canonical_dataframe(dataframe, mapping)


def run_sentiment_stage(
    canonical_df: pd.DataFrame,
    model_name: SentimentModelName | str,
    *,
    analyzer: SentimentAnalyzer | None = None,
) -> SentimentInferenceResult:
    """Run the completed sentiment stage against an existing canonical DataFrame."""
    selected = SentimentModelName(model_name)
    active_analyzer = analyzer or SentimentAnalyzer.load(selected)
    if active_analyzer.model_name is not selected:
        raise ValueError("Provided analyzer does not match the requested sentiment model.")
    return active_analyzer.predict_dataframe(canonical_df)


def run_topic_stage(
    sentiment_df: pd.DataFrame,
    *,
    config: TopicModelConfig | None = None,
    topic_model: NMFTopicModel | None = None,
    stability_runs: int = 3,
) -> TopicModelResult:
    """Discover NMF topics and enrich sentiment results with topic columns."""
    required = {"review_id", "review_text", "clean_text", "sentiment_label", "sentiment_score"}
    missing = sorted(required.difference(sentiment_df.columns))
    if missing:
        raise ValueError(
            "Topic modeling requires sentiment-enriched canonical data. Missing: "
            + ", ".join(missing)
        )

    active_model = topic_model or NMFTopicModel(config or TopicModelConfig())
    if topic_model is not None and config is not None and topic_model.config != config:
        raise ValueError("Provided topic model configuration does not match config.")

    result = active_model.fit_dataframe(sentiment_df)
    evaluation = evaluate_topic_model(
        active_model,
        result.dataframe,
        stability_runs=stability_runs,
    )
    # Keep the result dataclass focused on model outputs while attaching evaluation
    # as transparent metadata for persistence and UI consumers.
    result.model.training_metadata["evaluation"] = evaluation
    return result


def run_pipeline(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Run the future complete analysis pipeline.

    Ingestion, EDA, sentiment analysis, and topic modeling are implemented.
    Aspect analysis and insight generation remain deferred to later phases.
    """
    raise NotImplementedError(
        "The end-to-end pipeline is not complete yet. Sentiment and topic modeling "
        "are available through their stage functions; later phases remain pending."
    )
