"""Cautious rule-based recommendations grounded in analysis evidence."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.insights.trends import TrendAnalysisResult
from src.insights.utils import InsightRecommendation, PRIORITY_ORDER


RECOMMENDATION_COLUMNS = [
    "priority",
    "title",
    "affected_type",
    "affected_item",
    "supporting_metric",
    "supporting_value",
    "supporting_count",
    "explanation",
    "representative_review_ids",
]


def _priority_for_negative_share(negative_share: float, supporting_count: int) -> str:
    if negative_share >= 0.60 and supporting_count >= 3:
        return "High"
    if negative_share >= 0.40:
        return "Medium"
    return "Low"


def generate_recommendations(
    metrics: dict[str, Any],
    *,
    trends: TrendAnalysisResult | None = None,
) -> pd.DataFrame:
    """Generate deterministic recommendations without making causal claims."""
    items: list[InsightRecommendation] = []

    pain = metrics.get("priority_improvement")
    if pain:
        share = float(pain["negative_share"])
        count = int(pain["mention_count"])
        items.append(
            InsightRecommendation(
                priority=_priority_for_negative_share(share, count),
                title=f'Investigate {pain["aspect"]} feedback',
                affected_type="Aspect",
                affected_item=str(pain["aspect"]),
                supporting_metric="Negative mention share",
                supporting_value=share,
                supporting_count=count,
                explanation=(
                    f'{share:.1%} of {count:,} observed mentions for "{pain["aspect"]}" '
                    "are negative. Review the supporting feedback and investigate potential "
                    "product or operational causes before deciding on a change."
                ),
                representative_review_ids=tuple(pain.get("representative_review_ids", ())),
            )
        )

    strength = metrics.get("key_strength")
    if strength:
        share = float(strength["positive_share"])
        count = int(strength["mention_count"])
        items.append(
            InsightRecommendation(
                priority="Low",
                title=f'Protect the {strength["aspect"]} strength',
                affected_type="Aspect",
                affected_item=str(strength["aspect"]),
                supporting_metric="Positive mention share",
                supporting_value=share,
                supporting_count=count,
                explanation=(
                    f'{share:.1%} of {count:,} observed mentions for "{strength["aspect"]}" '
                    "are positive. Preserve the behaviors or product qualities associated with "
                    "this strength while validating them with representative reviews."
                ),
                representative_review_ids=tuple(strength.get("representative_review_ids", ())),
            )
        )

    if trends and trends.available and trends.worsening_aspect:
        item = trends.worsening_aspect
        affected = str(item["aspect"])
        if not any(rec.affected_item == affected and rec.priority == "High" for rec in items):
            change = float(item["negative_share_change"])
            count = int(item["supporting_count"])
            items.append(
                InsightRecommendation(
                    priority="High" if change >= 0.20 and count >= 3 else "Medium",
                    title=f'Monitor worsening {affected} feedback',
                    affected_type="Aspect",
                    affected_item=affected,
                    supporting_metric="Recent negative-share change",
                    supporting_value=change,
                    supporting_count=count,
                    explanation=(
                        f'Negative share for "{affected}" rose by {change:.1%} percentage '
                        "points between the two latest analysis periods. Investigate recent "
                        "changes and continue monitoring; the observed association does not "
                        "establish a cause."
                    ),
                    representative_review_ids=(),
                )
            )

    if not items:
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)

    dataframe = pd.DataFrame([item.as_dict() for item in items], columns=RECOMMENDATION_COLUMNS)
    dataframe["_priority_order"] = dataframe["priority"].map(PRIORITY_ORDER).fillna(99)
    dataframe = dataframe.sort_values(
        ["_priority_order", "supporting_count", "title"],
        ascending=[True, False, True],
    ).drop(columns="_priority_order")
    return dataframe.reset_index(drop=True)
