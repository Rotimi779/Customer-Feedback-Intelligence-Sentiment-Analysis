"""Heuristics for review-text and optional metadata column detection."""

from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.ingestion.schema import (
    DEFAULT_INGESTION_CONFIG,
    METADATA_ALIASES,
    TEXT_COLUMN_KEYWORDS,
    IngestionConfig,
)


_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class TextColumnCandidate:
    """Scored evidence that a source column contains customer feedback text."""

    column: str
    score: float
    average_length: float
    non_empty_ratio: float
    alphabetic_ratio: float
    unique_ratio: float
    keyword_score: float


def _normalized_name(name: object) -> str:
    normalized = _NORMALIZE_PATTERN.sub("_", str(name).strip().lower()).strip("_")
    return normalized or "column"


def _name_keyword_score(column: str) -> float:
    name = _normalized_name(column)
    tokens = set(name.split("_"))

    if name in TEXT_COLUMN_KEYWORDS:
        return 1.0
    if any(keyword in tokens for keyword in TEXT_COLUMN_KEYWORDS):
        return 0.9
    if any(keyword in name for keyword in TEXT_COLUMN_KEYWORDS):
        return 0.7
    return 0.0


def _sample_non_empty_text(
    series: pd.Series,
    *,
    max_rows: int,
) -> tuple[pd.Series, int]:
    sampled = series.head(max_rows)
    non_null = sampled[sampled.notna()]
    as_text = non_null.astype("string").str.strip()
    non_empty = as_text[as_text.ne("")]
    return non_empty, len(sampled)


def score_text_column(
    series: pd.Series,
    column_name: str,
    *,
    config: IngestionConfig = DEFAULT_INGESTION_CONFIG,
) -> TextColumnCandidate | None:
    """Score one column using name, content length, coverage, and uniqueness."""
    values, sample_size = _sample_non_empty_text(
        series,
        max_rows=config.detection_sample_rows,
    )
    if sample_size == 0 or values.empty:
        return None

    lengths = values.str.len()
    average_length = float(lengths.mean())
    non_empty_ratio = float(len(values) / sample_size)
    alphabetic_ratio = float(values.str.contains(r"[A-Za-z]", regex=True).mean())
    unique_ratio = float(values.nunique(dropna=True) / len(values))
    keyword_score = _name_keyword_score(column_name)

    # Unknown columns need stronger textual evidence. Keyword-named columns may
    # contain legitimate short feedback such as "Good" or "Slow".
    if alphabetic_ratio < 0.30:
        return None
    if keyword_score == 0.0 and average_length < 15.0:
        return None
    if keyword_score > 0.0 and average_length < 2.0:
        return None

    length_score = min(average_length / 80.0, 1.0)
    score = (
        0.40 * keyword_score
        + 0.25 * length_score
        + 0.15 * non_empty_ratio
        + 0.10 * alphabetic_ratio
        + 0.10 * unique_ratio
    )

    return TextColumnCandidate(
        column=column_name,
        score=round(float(score), 6),
        average_length=average_length,
        non_empty_ratio=non_empty_ratio,
        alphabetic_ratio=alphabetic_ratio,
        unique_ratio=unique_ratio,
        keyword_score=keyword_score,
    )


def rank_text_columns(
    dataframe: pd.DataFrame,
    *,
    config: IngestionConfig = DEFAULT_INGESTION_CONFIG,
) -> tuple[TextColumnCandidate, ...]:
    """Return viable text columns from strongest to weakest candidate."""
    candidates: list[TextColumnCandidate] = []

    for column in dataframe.columns:
        candidate = score_text_column(
            dataframe[column],
            str(column),
            config=config,
        )
        if candidate is not None:
            candidates.append(candidate)

    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.score,
                item.keyword_score,
                item.average_length,
            ),
            reverse=True,
        )
    )


def detect_text_column(
    dataframe: pd.DataFrame,
    *,
    config: IngestionConfig = DEFAULT_INGESTION_CONFIG,
) -> str | None:
    """Suggest the highest-ranked review-text column, when confidence is usable."""
    candidates = rank_text_columns(dataframe, config=config)
    if not candidates or candidates[0].score < config.minimum_text_score:
        return None
    return candidates[0].column


def _alias_match_score(column: str, aliases: tuple[str, ...]) -> int:
    name = _normalized_name(column)
    tokens = set(name.split("_"))

    if name in aliases:
        return 3
    if any(alias in tokens for alias in aliases):
        return 2
    if any(alias in name for alias in aliases if len(alias) >= 4):
        return 1
    return 0


def _is_compatible_metadata(series: pd.Series, canonical_name: str) -> bool:
    sample = series.dropna().head(500)
    if sample.empty:
        return False

    if canonical_name == "date":
        parsed = pd.to_datetime(sample, errors="coerce")
        return float(parsed.notna().mean()) >= 0.60

    if canonical_name == "rating":
        if is_numeric_dtype(sample):
            return True
        parsed = pd.to_numeric(sample, errors="coerce")
        return float(parsed.notna().mean()) >= 0.60

    text = sample.astype("string").str.strip()
    return bool(text.ne("").mean() >= 0.50)


def detect_metadata_columns(
    dataframe: pd.DataFrame,
    *,
    text_column: str | None = None,
) -> dict[str, str]:
    """Detect optional metadata columns without making them required."""
    detected: dict[str, str] = {}
    used_columns = {text_column} if text_column is not None else set()

    for canonical_name, aliases in METADATA_ALIASES.items():
        ranked: list[tuple[int, str]] = []
        for column in dataframe.columns:
            source_column = str(column)
            if source_column in used_columns:
                continue
            score = _alias_match_score(source_column, aliases)
            if score and _is_compatible_metadata(
                dataframe[source_column], canonical_name
            ):
                ranked.append((score, source_column))

        if ranked:
            ranked.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
            selected = ranked[0][1]
            detected[canonical_name] = selected
            used_columns.add(selected)

    return detected
