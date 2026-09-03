"""Associate extracted aspects with existing review-level sentiment outputs."""

from __future__ import annotations

import math

import pandas as pd

from src.aspects.extraction import AspectExtractor
from src.aspects.utils import AspectAnalysisError, SENTIMENT_TO_SCORE


REQUIRED_SENTIMENT_COLUMNS = {"sentiment_label"}


def _choose_text_column(dataframe: pd.DataFrame) -> str:
    for candidate in ("clean_text", "review_text"):
        if candidate in dataframe.columns:
            return candidate
    raise AspectAnalysisError(
        "Aspect extraction requires a 'clean_text' or 'review_text' column."
    )


def _canonical_sentiment(value: object) -> str:
    label = str(value).strip().title()
    if label not in SENTIMENT_TO_SCORE:
        raise AspectAnalysisError(
            "Aspect sentiment requires canonical review sentiment labels: "
            "Negative, Neutral, or Positive."
        )
    return label


def associate_aspect_sentiment(
    dataframe: pd.DataFrame,
    *,
    extractor: AspectExtractor | None = None,
) -> pd.DataFrame:
    """Return a non-mutating DataFrame enriched with aspect-level structures.

    The MVP intentionally reuses review-level sentiment for every detected aspect,
    matching the experiment specification. It does not claim clause-level or
    aspect-specific sentiment classification.
    """
    missing = sorted(REQUIRED_SENTIMENT_COLUMNS.difference(dataframe.columns))
    if missing:
        raise AspectAnalysisError(
            "Aspect analysis requires review-level sentiment first. Missing: "
            + ", ".join(missing)
        )

    text_column = _choose_text_column(dataframe)
    active_extractor = extractor or AspectExtractor()
    enriched = dataframe.copy(deep=True)

    detected_aspects: list[list[str]] = []
    aspect_sentiments: list[dict[str, str]] = []
    aspect_confidences: list[dict[str, float]] = []
    matched_terms_by_row: list[dict[str, list[str]]] = []

    has_confidence = "sentiment_score" in dataframe.columns

    for row in dataframe.itertuples(index=False):
        row_data = row._asdict()
        matches = active_extractor.extract(row_data[text_column])
        aspects = [match.aspect for match in matches]
        sentiment = _canonical_sentiment(row_data["sentiment_label"])

        detected_aspects.append(aspects)
        aspect_sentiments.append({aspect: sentiment for aspect in aspects})
        matched_terms_by_row.append(
            {match.aspect: list(match.matched_terms) for match in matches}
        )

        confidence_map: dict[str, float] = {}
        if has_confidence:
            raw_score = row_data.get("sentiment_score")
            if raw_score is not None:
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    score = math.nan
                if not math.isnan(score):
                    confidence_map = {aspect: score for aspect in aspects}
        aspect_confidences.append(confidence_map)

    enriched["detected_aspects"] = detected_aspects
    enriched["aspect_sentiment"] = aspect_sentiments
    if has_confidence:
        enriched["aspect_confidence"] = aspect_confidences

    # Internal explainability field is intentionally not added to the public
    # canonical schema; callers can re-run the extractor when inspecting rules.
    return enriched
