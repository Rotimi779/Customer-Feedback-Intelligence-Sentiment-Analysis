"""Dataset-quality metrics for exploratory analysis.

The functions in this module are intentionally independent of Streamlit so
that they can be reused by dashboard pages, tests, and later pipeline stages.
They never mutate the supplied DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class DataQualitySummary:
    """High-level quality indicators for one DataFrame."""

    total_rows: int
    duplicate_reviews: int
    empty_reviews: int
    missing_cells: int
    total_cells: int

    @property
    def completeness_percentage(self) -> float:
        """Return the percentage of populated cells in the dataset."""
        if self.total_cells == 0:
            return 100.0
        return 100.0 * (1.0 - (self.missing_cells / self.total_cells))


def _blank_or_missing_mask(series: pd.Series) -> pd.Series:
    """Return a Boolean mask treating blank strings as missing values."""
    mask = series.isna()
    if pd.api.types.is_object_dtype(series.dtype) or isinstance(
        series.dtype,
        pd.StringDtype,
    ):
        mask = mask | series.astype("string").fillna("").str.strip().eq("")
    return mask


def missing_value_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and percentages for every source column.

    Empty or whitespace-only strings are counted as missing for text-like
    columns. The output keeps zero-missing columns so the dashboard can show a
    complete quality profile rather than hiding healthy columns.
    """
    rows = len(dataframe)
    records: list[dict[str, object]] = []

    for column in dataframe.columns:
        series = dataframe[column]
        missing_count = int(_blank_or_missing_mask(series).sum())
        percentage = 0.0 if rows == 0 else (missing_count / rows) * 100.0
        records.append(
            {
                "column": str(column),
                "dtype": str(series.dtype),
                "missing_count": missing_count,
                "missing_percentage": percentage,
            }
        )

    return pd.DataFrame(
        records,
        columns=(
            "column",
            "dtype",
            "missing_count",
            "missing_percentage",
        ),
    )


def count_empty_reviews(
    dataframe: pd.DataFrame,
    *,
    text_column: str = "review_text",
) -> int:
    """Count null, empty, or whitespace-only review values."""
    if text_column not in dataframe.columns:
        return 0
    return int(_blank_or_missing_mask(dataframe[text_column]).sum())


def count_duplicate_reviews(
    dataframe: pd.DataFrame,
    *,
    text_column: str = "clean_text",
) -> int:
    """Count repeated reviews after safe whitespace and case normalization."""
    if text_column not in dataframe.columns:
        return 0

    normalized = (
        dataframe[text_column]
        .astype("string")
        .fillna("")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.casefold()
    )
    usable = normalized.ne("")
    return int(normalized.loc[usable].duplicated(keep="first").sum())


def summarize_quality(
    dataframe: pd.DataFrame,
    *,
    text_column: str = "clean_text",
) -> DataQualitySummary:
    """Calculate a compact, reusable dataset-quality summary."""
    missing_table = missing_value_summary(dataframe)
    return DataQualitySummary(
        total_rows=len(dataframe),
        duplicate_reviews=count_duplicate_reviews(
            dataframe,
            text_column=text_column,
        ),
        empty_reviews=count_empty_reviews(
            dataframe,
            text_column=text_column,
        ),
        missing_cells=int(missing_table["missing_count"].sum()),
        total_cells=int(dataframe.shape[0] * dataframe.shape[1]),
    )
