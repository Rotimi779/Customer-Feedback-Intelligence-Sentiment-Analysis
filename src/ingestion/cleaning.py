"""Column normalization and canonical DataFrame construction."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from src.ingestion.schema import (
    OPTIONAL_CANONICAL_COLUMNS,
    REQUIRED_CANONICAL_COLUMNS,
    REVIEW_ID_ALIASES,
    CanonicalizationResult,
    ColumnMapping,
    IngestionStatistics,
    ValidationMessage,
    ValidationSeverity,
)


_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_REPEATED_UNDERSCORE = re.compile(r"_+")


def normalize_column_name(column: object) -> str:
    """Convert one source header into a stable snake_case identifier."""
    ascii_name = (
        unicodedata.normalize("NFKD", str(column))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    normalized = _NON_ALPHANUMERIC.sub("_", ascii_name.strip().lower())
    normalized = _REPEATED_UNDERSCORE.sub("_", normalized).strip("_")
    return normalized or "column"


def normalize_column_names(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return a copy with unique normalized headers and an audit mapping."""
    normalized_names: list[str] = []
    used_counts: dict[str, int] = {}
    audit_mapping: dict[str, str] = {}

    for original in dataframe.columns:
        base = normalize_column_name(original)
        used_counts[base] = used_counts.get(base, 0) + 1
        count = used_counts[base]
        normalized = base if count == 1 else f"{base}_{count}"
        normalized_names.append(normalized)
        audit_mapping[str(original)] = normalized

    normalized_dataframe = dataframe.copy()
    normalized_dataframe.columns = normalized_names
    return normalized_dataframe, audit_mapping


def normalize_review_text(series: pd.Series) -> pd.Series:
    """Apply only shared ingestion cleaning, not model-specific preprocessing."""
    return (
        series.astype("string")
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _source_review_ids(dataframe: pd.DataFrame) -> pd.Series | None:
    for column in REVIEW_ID_ALIASES:
        if column not in dataframe.columns:
            continue
        identifiers = dataframe[column].astype("string").str.strip()
        if identifiers.notna().all() and identifiers.ne("").all() and identifiers.is_unique:
            return identifiers
    return None


def _canonical_optional_series(
    dataframe: pd.DataFrame,
    canonical_name: str,
    source_column: str,
) -> tuple[pd.Series, ValidationMessage | None]:
    source = dataframe[source_column]

    if canonical_name == "date":
        converted = pd.to_datetime(source, errors="coerce")
        invalid_count = int(source.notna().sum() - converted.notna().sum())
        warning = None
        if invalid_count:
            warning = ValidationMessage(
                code="invalid_dates_coerced",
                message=(
                    f"{invalid_count:,} value(s) in '{source_column}' could not be "
                    "parsed as dates and were set to missing."
                ),
                severity=ValidationSeverity.WARNING,
            )
        return converted, warning

    if canonical_name == "rating":
        converted = pd.to_numeric(source, errors="coerce")
        invalid_count = int(source.notna().sum() - converted.notna().sum())
        warning = None
        if invalid_count:
            warning = ValidationMessage(
                code="invalid_ratings_coerced",
                message=(
                    f"{invalid_count:,} value(s) in '{source_column}' could not be "
                    "parsed as numeric ratings and were set to missing."
                ),
                severity=ValidationSeverity.WARNING,
            )
        return converted, warning

    converted = source.astype("string").str.strip().replace("", pd.NA)
    return converted, None


def build_canonical_dataframe(
    dataframe: pd.DataFrame,
    mapping: ColumnMapping,
) -> CanonicalizationResult:
    """Remove unusable rows and return the canonical ingestion data contract.

    ``review_text`` preserves the selected source values. ``clean_text`` only
    trims and normalizes whitespace. Lowercasing, punctuation handling, and
    model-specific processing belong to later phases.
    """
    if mapping.text not in dataframe.columns:
        raise KeyError(f"Text column '{mapping.text}' does not exist.")

    input_rows = len(dataframe)
    source_text = dataframe[mapping.text].astype("string")
    clean_text = normalize_review_text(source_text)

    non_empty_mask = clean_text.ne("")
    empty_reviews_removed = int((~non_empty_mask).sum())

    working = dataframe.loc[non_empty_mask].copy()
    working_source_text = source_text.loc[non_empty_mask]
    working_clean_text = clean_text.loc[non_empty_mask]

    duplicate_mask = working_clean_text.str.casefold().duplicated(keep="first")
    duplicate_reviews_removed = int(duplicate_mask.sum())

    working = working.loc[~duplicate_mask].copy()
    working_source_text = working_source_text.loc[~duplicate_mask]
    working_clean_text = working_clean_text.loc[~duplicate_mask]

    source_ids = _source_review_ids(working)
    if source_ids is None:
        source_ids = pd.Series(
            [f"review_{index:06d}" for index in range(1, len(working) + 1)],
            index=working.index,
            dtype="string",
        )

    canonical = pd.DataFrame(index=working.index)
    canonical["review_id"] = source_ids.astype("string")
    canonical["review_text"] = working_source_text.astype("string")
    canonical["clean_text"] = working_clean_text.astype("string")

    warnings: list[ValidationMessage] = []
    for canonical_name, source_column in mapping.optional_items():
        if source_column is None:
            continue
        converted, warning = _canonical_optional_series(
            working,
            canonical_name,
            source_column,
        )
        canonical[canonical_name] = converted
        if warning is not None:
            warnings.append(warning)

    canonical = canonical.reset_index(drop=True)
    ordered_columns = list(REQUIRED_CANONICAL_COLUMNS) + [
        column for column in OPTIONAL_CANONICAL_COLUMNS if column in canonical.columns
    ]
    canonical = canonical.loc[:, ordered_columns]

    statistics = IngestionStatistics(
        input_rows=input_rows,
        output_rows=len(canonical),
        empty_reviews_removed=empty_reviews_removed,
        duplicate_reviews_removed=duplicate_reviews_removed,
    )
    return CanonicalizationResult(
        dataframe=canonical,
        statistics=statistics,
        warnings=tuple(warnings),
    )
