"""Application pipeline contracts and completed ingestion orchestration."""

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


def run_pipeline(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Run the complete analysis pipeline once later phases are implemented.

    The ingestion stage is available through :func:`prepare_dataset`, and EDA
    is exposed through the independent ``src.eda`` utilities. Sentiment, topics,
    aspects, and insights remain intentionally deferred.
    """
    raise NotImplementedError(
        "Only data ingestion is implemented. Later analysis phases are pending."
    )
