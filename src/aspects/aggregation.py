"""Aspect mention expansion, aggregation, and structural evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.aspects.extraction import AspectExtractor
from src.aspects.sentiment import associate_aspect_sentiment
from src.aspects.utils import AspectAnalysisError, SENTIMENT_TO_SCORE


@dataclass(frozen=True)
class AspectAnalysisResult:
    """Outputs produced by one deterministic aspect-analysis run."""

    dataframe: pd.DataFrame
    mentions: pd.DataFrame
    summary: pd.DataFrame
    evaluation: dict[str, Any]


def explode_aspect_mentions(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create one row per review-aspect mention for aggregation and charts."""
    required = {"detected_aspects", "aspect_sentiment"}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise AspectAnalysisError(
            "Aspect mention expansion requires enriched aspect columns. Missing: "
            + ", ".join(missing)
        )

    rows: list[dict[str, object]] = []
    passthrough = [
        column
        for column in (
            "review_id",
            "review_text",
            "clean_text",
            "sentiment_label",
            "sentiment_score",
            "topic_id",
            "topic_label",
            "product",
            "category",
            "date",
            "rating",
        )
        if column in dataframe.columns
    ]

    for _, row in dataframe.iterrows():
        aspects = row.get("detected_aspects")
        if not isinstance(aspects, (list, tuple)) or not aspects:
            continue
        sentiment_map = row.get("aspect_sentiment")
        sentiment_map = sentiment_map if isinstance(sentiment_map, dict) else {}
        confidence_map = row.get("aspect_confidence")
        confidence_map = confidence_map if isinstance(confidence_map, dict) else {}

        for aspect in aspects:
            label = str(sentiment_map.get(aspect, row.get("sentiment_label", ""))).title()
            payload = {column: row[column] for column in passthrough}
            payload.update(
                {
                    "aspect": str(aspect),
                    "aspect_sentiment_label": label,
                    "aspect_sentiment_numeric": SENTIMENT_TO_SCORE.get(label),
                    "aspect_confidence": confidence_map.get(aspect),
                }
            )
            rows.append(payload)

    return pd.DataFrame(rows)


def build_aspect_summary(
    mentions: pd.DataFrame,
    *,
    total_reviews: int,
) -> pd.DataFrame:
    """Aggregate aspect frequency, sentiment, confidence, and optional ratings."""
    columns = [
        "aspect",
        "mention_count",
        "unique_review_count",
        "review_coverage",
        "average_sentiment",
        "dominant_sentiment",
        "positive_count",
        "neutral_count",
        "negative_count",
        "positive_share",
        "neutral_share",
        "negative_share",
        "average_confidence",
        "average_rating",
    ]
    if mentions.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for aspect, subset in mentions.groupby("aspect", sort=True, observed=True):
        sentiment_counts = (
            subset["aspect_sentiment_label"]
            .value_counts()
            .reindex(["Negative", "Neutral", "Positive"], fill_value=0)
        )
        mention_count = int(len(subset))
        unique_review_count = (
            int(subset["review_id"].astype(str).nunique())
            if "review_id" in subset.columns
            else mention_count
        )
        numeric = pd.to_numeric(
            subset["aspect_sentiment_numeric"], errors="coerce"
        ).dropna()
        dominant_sentiment = str(sentiment_counts.idxmax())

        confidence = (
            pd.to_numeric(subset["aspect_confidence"], errors="coerce").dropna()
            if "aspect_confidence" in subset.columns
            else pd.Series(dtype=float)
        )
        rating = (
            pd.to_numeric(subset["rating"], errors="coerce").dropna()
            if "rating" in subset.columns
            else pd.Series(dtype=float)
        )

        rows.append(
            {
                "aspect": str(aspect),
                "mention_count": mention_count,
                "unique_review_count": unique_review_count,
                "review_coverage": (
                    float(unique_review_count / total_reviews) if total_reviews else 0.0
                ),
                "average_sentiment": float(numeric.mean()) if not numeric.empty else None,
                "dominant_sentiment": dominant_sentiment,
                "positive_count": int(sentiment_counts["Positive"]),
                "neutral_count": int(sentiment_counts["Neutral"]),
                "negative_count": int(sentiment_counts["Negative"]),
                "positive_share": float(sentiment_counts["Positive"] / mention_count),
                "neutral_share": float(sentiment_counts["Neutral"] / mention_count),
                "negative_share": float(sentiment_counts["Negative"] / mention_count),
                "average_confidence": (
                    float(confidence.mean()) if not confidence.empty else None
                ),
                "average_rating": float(rating.mean()) if not rating.empty else None,
            }
        )

    return pd.DataFrame(rows, columns=columns).sort_values(
        ["mention_count", "aspect"], ascending=[False, True]
    ).reset_index(drop=True)


def evaluate_aspect_outputs(
    dataframe: pd.DataFrame,
    mentions: pd.DataFrame,
) -> dict[str, Any]:
    """Return measurable structural metrics without inventing gold-label accuracy."""
    total_reviews = int(len(dataframe))
    reviews_with_aspects = int(
        dataframe["detected_aspects"].apply(
            lambda value: isinstance(value, (list, tuple)) and len(value) > 0
        ).sum()
    )
    multi_aspect_reviews = int(
        dataframe["detected_aspects"].apply(
            lambda value: isinstance(value, (list, tuple)) and len(value) > 1
        ).sum()
    )
    sentiment_complete = bool(
        mentions.empty
        or mentions["aspect_sentiment_label"].isin(SENTIMENT_TO_SCORE).all()
    )

    return {
        "total_reviews": total_reviews,
        "reviews_with_aspects": reviews_with_aspects,
        "aspect_coverage": (
            float(reviews_with_aspects / total_reviews) if total_reviews else 0.0
        ),
        "total_aspect_mentions": int(len(mentions)),
        "unique_aspects": int(mentions["aspect"].nunique()) if not mentions.empty else 0,
        "multi_aspect_reviews": multi_aspect_reviews,
        "multi_aspect_review_share": (
            float(multi_aspect_reviews / total_reviews) if total_reviews else 0.0
        ),
        "sentiment_association_complete": sentiment_complete,
        "manual_review_required": True,
        "manual_review_note": (
            "Correct aspect detection, label relevance, and business usefulness require "
            "manual inspection because the MVP has no gold aspect-label dataset."
        ),
    }


def analyze_aspects(
    dataframe: pd.DataFrame,
    *,
    extractor: AspectExtractor | None = None,
) -> AspectAnalysisResult:
    """Run extraction, review-sentiment reuse, aggregation, and evaluation."""
    enriched = associate_aspect_sentiment(dataframe, extractor=extractor)
    mentions = explode_aspect_mentions(enriched)
    summary = build_aspect_summary(mentions, total_reviews=len(enriched))
    evaluation = evaluate_aspect_outputs(enriched, mentions)
    return AspectAnalysisResult(
        dataframe=enriched,
        mentions=mentions,
        summary=summary,
        evaluation=evaluation,
    )
