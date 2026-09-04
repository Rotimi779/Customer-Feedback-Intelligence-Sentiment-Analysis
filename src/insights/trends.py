"""Optional date-based trend analysis for evidence-backed business insights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.eda.statistics import infer_time_frequency
from src.insights.utils import SENTIMENT_TO_SCORE, canonical_sentiment_series, minimum_evidence_support


@dataclass(frozen=True)
class TrendAnalysisResult:
    """Reusable trend tables and carefully scoped change signals."""

    available: bool
    reason: str | None
    frequency: str | None
    review_volume: pd.DataFrame
    sentiment: pd.DataFrame
    topics: pd.DataFrame
    aspects: pd.DataFrame
    fastest_growing_topic: dict[str, Any] | None
    worsening_aspect: dict[str, Any] | None
    sentiment_change: dict[str, Any] | None


def _empty(reason: str) -> TrendAnalysisResult:
    return TrendAnalysisResult(
        available=False,
        reason=reason,
        frequency=None,
        review_volume=pd.DataFrame(columns=["period", "review_count"]),
        sentiment=pd.DataFrame(columns=["period", "sentiment_label", "review_count", "share"]),
        topics=pd.DataFrame(columns=["period", "topic_label", "review_count", "share"]),
        aspects=pd.DataFrame(
            columns=["period", "aspect", "mention_count", "negative_count", "negative_share"]
        ),
        fastest_growing_topic=None,
        worsening_aspect=None,
        sentiment_change=None,
    )


def _prepare_dates(dataframe: pd.DataFrame, frequency: str) -> pd.DataFrame:
    dated = dataframe.copy()
    dated["_analysis_date"] = pd.to_datetime(dated["date"], errors="coerce")
    dated = dated.loc[dated["_analysis_date"].notna()].copy()
    dated["period"] = dated["_analysis_date"].dt.to_period(frequency).dt.to_timestamp()
    return dated


def _latest_two_periods(frame: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if frame.empty or "period" not in frame.columns:
        return None
    periods = sorted(pd.to_datetime(frame["period"].dropna().unique()).tolist())
    if len(periods) < 2:
        return None
    return periods[-2], periods[-1]


def _build_sentiment_trend(dated: pd.DataFrame) -> pd.DataFrame:
    if "sentiment_label" not in dated.columns or dated.empty:
        return pd.DataFrame(columns=["period", "sentiment_label", "review_count", "share"])
    work = dated[["period", "sentiment_label"]].copy()
    work["sentiment_label"] = canonical_sentiment_series(work["sentiment_label"])
    counts = (
        work.groupby(["period", "sentiment_label"], observed=True)
        .size()
        .reset_index(name="review_count")
    )
    totals = counts.groupby("period", observed=True)["review_count"].transform("sum")
    counts["share"] = counts["review_count"] / totals
    return counts.sort_values(["period", "sentiment_label"]).reset_index(drop=True)


def _build_topic_trend(dated: pd.DataFrame) -> pd.DataFrame:
    if "topic_label" not in dated.columns or dated.empty:
        return pd.DataFrame(columns=["period", "topic_label", "review_count", "share"])
    work = dated.loc[dated["topic_label"].notna(), ["period", "topic_label"]].copy()
    if work.empty:
        return pd.DataFrame(columns=["period", "topic_label", "review_count", "share"])
    work["topic_label"] = work["topic_label"].astype(str)
    counts = (
        work.groupby(["period", "topic_label"], observed=True)
        .size()
        .reset_index(name="review_count")
    )
    totals = counts.groupby("period", observed=True)["review_count"].transform("sum")
    counts["share"] = counts["review_count"] / totals
    return counts.sort_values(["period", "topic_label"]).reset_index(drop=True)


def _build_aspect_trend(
    mentions: pd.DataFrame | None,
    *,
    frequency: str,
) -> pd.DataFrame:
    columns = ["period", "aspect", "mention_count", "negative_count", "negative_share"]
    if mentions is None or mentions.empty or "date" not in mentions.columns:
        return pd.DataFrame(columns=columns)

    work = mentions.copy()
    work["_analysis_date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.loc[work["_analysis_date"].notna() & work["aspect"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)
    work["period"] = work["_analysis_date"].dt.to_period(frequency).dt.to_timestamp()
    work["is_negative"] = work["aspect_sentiment_label"].astype(str).str.title().eq("Negative")
    grouped = (
        work.groupby(["period", "aspect"], observed=True)
        .agg(mention_count=("aspect", "size"), negative_count=("is_negative", "sum"))
        .reset_index()
    )
    grouped["negative_count"] = grouped["negative_count"].astype(int)
    grouped["negative_share"] = grouped["negative_count"] / grouped["mention_count"]
    return grouped[columns].sort_values(["period", "aspect"]).reset_index(drop=True)


def _fastest_growing_topic(
    topic_trend: pd.DataFrame,
    *,
    total_reviews: int,
) -> dict[str, Any] | None:
    pair = _latest_two_periods(topic_trend)
    if pair is None:
        return None
    previous_period, recent_period = pair
    previous = topic_trend.loc[topic_trend["period"].eq(previous_period)].set_index("topic_label")
    recent = topic_trend.loc[topic_trend["period"].eq(recent_period)].set_index("topic_label")
    support = minimum_evidence_support(total_reviews)

    records: list[dict[str, Any]] = []
    for topic in sorted(set(previous.index).union(recent.index)):
        previous_share = float(previous.at[topic, "share"]) if topic in previous.index else 0.0
        recent_share = float(recent.at[topic, "share"]) if topic in recent.index else 0.0
        recent_count = int(recent.at[topic, "review_count"]) if topic in recent.index else 0
        if recent_count < support:
            continue
        records.append(
            {
                "topic_label": str(topic),
                "previous_period": previous_period,
                "recent_period": recent_period,
                "previous_share": previous_share,
                "recent_share": recent_share,
                "share_change": recent_share - previous_share,
                "supporting_count": recent_count,
            }
        )
    if not records:
        return None
    best = max(records, key=lambda item: (item["share_change"], item["supporting_count"]))
    return best if float(best["share_change"]) > 0 else None


def _worsening_aspect(
    aspect_trend: pd.DataFrame,
    *,
    total_reviews: int,
) -> dict[str, Any] | None:
    pair = _latest_two_periods(aspect_trend)
    if pair is None:
        return None
    previous_period, recent_period = pair
    previous = aspect_trend.loc[aspect_trend["period"].eq(previous_period)].set_index("aspect")
    recent = aspect_trend.loc[aspect_trend["period"].eq(recent_period)].set_index("aspect")
    support = minimum_evidence_support(total_reviews)

    records: list[dict[str, Any]] = []
    for aspect in sorted(set(previous.index).intersection(recent.index)):
        previous_count = int(previous.at[aspect, "mention_count"])
        recent_count = int(recent.at[aspect, "mention_count"])
        if recent_count < support or previous_count < 1:
            continue
        previous_share = float(previous.at[aspect, "negative_share"])
        recent_share = float(recent.at[aspect, "negative_share"])
        records.append(
            {
                "aspect": str(aspect),
                "previous_period": previous_period,
                "recent_period": recent_period,
                "previous_negative_share": previous_share,
                "recent_negative_share": recent_share,
                "negative_share_change": recent_share - previous_share,
                "supporting_count": recent_count,
            }
        )
    if not records:
        return None
    best = max(
        records,
        key=lambda item: (item["negative_share_change"], item["supporting_count"]),
    )
    return best if float(best["negative_share_change"]) > 0 else None


def _sentiment_change(sentiment_trend: pd.DataFrame) -> dict[str, Any] | None:
    pair = _latest_two_periods(sentiment_trend)
    if pair is None:
        return None
    previous_period, recent_period = pair

    def negative_share(period: pd.Timestamp) -> float:
        subset = sentiment_trend.loc[sentiment_trend["period"].eq(period)]
        match = subset.loc[subset["sentiment_label"].eq("Negative"), "share"]
        return float(match.iloc[0]) if not match.empty else 0.0

    previous = negative_share(previous_period)
    recent = negative_share(recent_period)
    return {
        "previous_period": previous_period,
        "recent_period": recent_period,
        "previous_negative_share": previous,
        "recent_negative_share": recent,
        "negative_share_change": recent - previous,
    }


def analyze_trends(
    dataframe: pd.DataFrame,
    *,
    aspect_mentions: pd.DataFrame | None = None,
) -> TrendAnalysisResult:
    """Calculate optional trends without implying causality from observed feedback."""
    if "date" not in dataframe.columns:
        return _empty("Trend analysis requires a mapped date column.")

    dates = pd.to_datetime(dataframe["date"], errors="coerce").dropna()
    if dates.empty:
        return _empty("The mapped date column contains no usable dates.")
    if dates.nunique() < 2:
        return _empty("At least two distinct dates are required for trend analysis.")

    frequency = infer_time_frequency(dataframe, date_column="date")
    dated = _prepare_dates(dataframe, frequency)
    volume = (
        dated.groupby("period", observed=True)
        .size()
        .rename("review_count")
        .reset_index()
        .sort_values("period")
        .reset_index(drop=True)
    )
    sentiment = _build_sentiment_trend(dated)
    topics = _build_topic_trend(dated)
    aspects = _build_aspect_trend(aspect_mentions, frequency=frequency)

    if len(volume) < 2:
        return TrendAnalysisResult(
            available=False,
            reason="The available dates collapse into fewer than two analysis periods.",
            frequency=frequency,
            review_volume=volume,
            sentiment=sentiment,
            topics=topics,
            aspects=aspects,
            fastest_growing_topic=None,
            worsening_aspect=None,
            sentiment_change=None,
        )

    return TrendAnalysisResult(
        available=True,
        reason=None,
        frequency=frequency,
        review_volume=volume,
        sentiment=sentiment,
        topics=topics,
        aspects=aspects,
        fastest_growing_topic=_fastest_growing_topic(topics, total_reviews=len(dataframe)),
        worsening_aspect=_worsening_aspect(aspects, total_reviews=len(dataframe)),
        sentiment_change=_sentiment_change(sentiment),
    )
