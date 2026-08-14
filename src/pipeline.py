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


def run_pipeline(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Run the future complete analysis pipeline.

    Ingestion, model-free EDA, and sentiment inference are implemented. Topic
    modeling, aspect analysis, and insight generation remain intentionally
    deferred to their assigned phases.
    """
    raise NotImplementedError(
        "The end-to-end pipeline is not complete yet. Sentiment is available "
        "through run_sentiment_stage; later NLP phases remain pending."
    )
