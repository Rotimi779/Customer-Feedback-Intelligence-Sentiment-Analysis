"""Deterministic executive-summary generation from measured analysis outputs."""

from __future__ import annotations

from typing import Any

from src.insights.trends import TrendAnalysisResult


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.1%}"


def generate_executive_summary(
    metrics: dict[str, Any],
    *,
    trends: TrendAnalysisResult | None = None,
) -> str:
    """Create a concise, non-generative summary using only measured results."""
    total = int(metrics.get("total_reviews", 0))
    sentiment = metrics.get("sentiment", {})
    dominant = str(sentiment.get("dominant_sentiment") or "Unavailable")
    positive_share = sentiment.get("positive_share")
    neutral_share = sentiment.get("neutral_share")
    negative_share = sentiment.get("negative_share")

    sentences = [
        (
            f"Analyzed {total:,} reviews. Overall sentiment is {dominant}, with "
            f"{_percent(positive_share)} positive, {_percent(neutral_share)} neutral, "
            f"and {_percent(negative_share)} negative feedback."
        )
    ]

    topic = metrics.get("most_discussed_topic")
    if topic:
        sentences.append(
            f'The most discussed topic is "{topic["topic_label"]}" '
            f'({int(topic["review_count"]):,} reviews; {_percent(topic["share"])} of the dataset).'
        )

    pain = metrics.get("priority_improvement")
    if pain:
        sentences.append(
            f'The largest current pain point is "{pain["aspect"]}": '
            f'{_percent(pain["negative_share"])} of its {int(pain["mention_count"]):,} '
            "mentions are negative."
        )

    strength = metrics.get("key_strength")
    if strength:
        sentences.append(
            f'A notable strength is "{strength["aspect"]}": '
            f'{_percent(strength["positive_share"])} of its {int(strength["mention_count"]):,} '
            "mentions are positive."
        )

    if trends and trends.available and trends.worsening_aspect:
        item = trends.worsening_aspect
        sentences.append(
            f'Recently, negative share for "{item["aspect"]}" increased by '
            f'{float(item["negative_share_change"]) * 100:.1f} percentage points between the '
            "two latest analysis periods; this is an observed feedback pattern, not evidence of causation."
        )

    return " ".join(sentences)
