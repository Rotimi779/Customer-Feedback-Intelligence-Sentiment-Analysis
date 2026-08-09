"""Dataset validation with plain-language, actionable messages."""

from __future__ import annotations

import pandas as pd

from src.ingestion.column_detector import score_text_column
from src.ingestion.schema import (
    DEFAULT_INGESTION_CONFIG,
    ColumnMapping,
    IngestionConfig,
    ValidationReport,
    ValidationSeverity,
)


class DatasetValidationError(ValueError):
    """Raised when canonicalization is requested for an invalid dataset."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        summary = "; ".join(message.message for message in report.errors)
        super().__init__(summary or "Dataset validation failed.")


def validate_dataframe(
    dataframe: pd.DataFrame,
    *,
    config: IngestionConfig = DEFAULT_INGESTION_CONFIG,
) -> ValidationReport:
    """Validate dataset-level conditions that do not depend on column mapping."""
    report = ValidationReport()

    if dataframe.empty:
        report.add(
            code="empty_dataset",
            message="The dataset has no data rows.",
            severity=ValidationSeverity.ERROR,
            remediation="Add customer feedback rows and upload the CSV again.",
        )
        return report

    if len(dataframe.columns) == 0:
        report.add(
            code="missing_columns",
            message="The dataset has no readable columns.",
            severity=ValidationSeverity.ERROR,
            remediation="Add a header row with at least one feedback column.",
        )
        return report

    if len(dataframe) > config.max_rows:
        report.add(
            code="row_limit_exceeded",
            message=(
                f"The dataset contains more than the supported "
                f"{config.max_rows:,} rows."
            ),
            severity=ValidationSeverity.ERROR,
            remediation="Upload a representative sample of the dataset.",
        )

    entirely_empty_columns = tuple(
        str(column) for column in dataframe.columns if dataframe[column].isna().all()
    )
    if entirely_empty_columns:
        report.add(
            code="empty_columns",
            message=(
                f"{len(entirely_empty_columns)} column(s) contain only missing values: "
                + ", ".join(entirely_empty_columns[:5])
            ),
            severity=ValidationSeverity.WARNING,
            remediation="These columns can be removed from the source CSV if unused.",
        )

    return report


def validate_text_column(
    dataframe: pd.DataFrame,
    text_column: str,
    *,
    config: IngestionConfig = DEFAULT_INGESTION_CONFIG,
) -> ValidationReport:
    """Validate the manually selected or automatically suggested text column."""
    report = ValidationReport()

    if text_column not in dataframe.columns:
        report.add(
            code="text_column_missing",
            message=f"The selected text column '{text_column}' does not exist.",
            severity=ValidationSeverity.ERROR,
            remediation="Choose one of the columns shown in the uploaded dataset.",
        )
        return report

    values = dataframe[text_column].astype("string")
    clean_values = values.fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    usable_count = int(clean_values.ne("").sum())

    if usable_count == 0:
        report.add(
            code="text_column_empty",
            message=f"The selected column '{text_column}' contains no usable reviews.",
            severity=ValidationSeverity.ERROR,
            remediation="Choose a populated customer-feedback column.",
        )
        return report

    candidate = score_text_column(
        dataframe[text_column],
        text_column,
        config=config,
    )
    if candidate is None:
        report.add(
            code="text_column_not_text_like",
            message=(
                f"The selected column '{text_column}' does not appear to contain "
                "customer feedback text."
            ),
            severity=ValidationSeverity.ERROR,
            remediation=(
                "Choose a column containing written reviews, comments, or messages."
            ),
        )
        return report

    missing_count = len(dataframe) - usable_count
    if missing_count:
        report.add(
            code="empty_reviews_removed",
            message=(
                f"{missing_count:,} row(s) have empty values in '{text_column}' and "
                "will be removed."
            ),
            severity=ValidationSeverity.WARNING,
        )

    duplicate_count = int(clean_values[clean_values.ne("")].str.casefold().duplicated().sum())
    if duplicate_count:
        report.add(
            code="duplicate_reviews_removed",
            message=(
                f"{duplicate_count:,} duplicate review(s) will be removed using "
                "normalized review text."
            ),
            severity=ValidationSeverity.WARNING,
        )

    populated_ratio = usable_count / len(dataframe)
    if populated_ratio < config.minimum_non_empty_ratio:
        report.add(
            code="low_text_coverage",
            message=(
                f"Only {populated_ratio:.1%} of rows contain usable review text in "
                f"'{text_column}'."
            ),
            severity=ValidationSeverity.ERROR,
            remediation="Choose a more complete feedback column.",
        )

    return report


def validate_column_mapping(
    dataframe: pd.DataFrame,
    mapping: ColumnMapping,
) -> ValidationReport:
    """Ensure optional mappings point to real, non-conflicting source columns."""
    report = ValidationReport()
    selected: dict[str, str] = {"text": mapping.text}

    for canonical_name, source_column in mapping.optional_items():
        if source_column is None:
            continue
        if source_column not in dataframe.columns:
            report.add(
                code="mapped_column_missing",
                message=(
                    f"The mapped {canonical_name} column '{source_column}' does not "
                    "exist in the dataset."
                ),
                severity=ValidationSeverity.ERROR,
                remediation="Select a valid source column or leave this field unmapped.",
            )
            continue
        if source_column in selected.values():
            existing_name = next(
                name for name, value in selected.items() if value == source_column
            )
            report.add(
                code="duplicate_column_mapping",
                message=(
                    f"'{source_column}' is mapped to both {existing_name} and "
                    f"{canonical_name}."
                ),
                severity=ValidationSeverity.ERROR,
                remediation="Map each source column to only one internal field.",
            )
            continue
        selected[canonical_name] = source_column

    return report


def validate_dataset(
    dataframe: pd.DataFrame,
    mapping: ColumnMapping,
    *,
    config: IngestionConfig = DEFAULT_INGESTION_CONFIG,
) -> ValidationReport:
    """Run all pre-canonicalization validation rules."""
    report = validate_dataframe(dataframe, config=config)
    if not report.is_valid:
        return report

    report.extend(validate_text_column(dataframe, mapping.text, config=config))
    report.extend(validate_column_mapping(dataframe, mapping))
    return report
