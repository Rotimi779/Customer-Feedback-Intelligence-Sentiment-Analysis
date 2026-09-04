"""Export finalized Phase 7 results without persisting uploaded customer data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.insights.generator import BusinessInsightsResult


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Serialize a finalized DataFrame to UTF-8 CSV bytes."""
    return dataframe.to_csv(index=False).encode("utf-8")


def recommendations_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    """Serialize recommendations while flattening evidence review IDs."""
    output = dataframe.copy()
    if "representative_review_ids" in output.columns:
        output["representative_review_ids"] = output["representative_review_ids"].apply(
            lambda value: ", ".join(str(item) for item in value)
            if isinstance(value, (list, tuple))
            else str(value)
        )
    return dataframe_to_csv_bytes(output)


def build_markdown_report(result: "BusinessInsightsResult") -> str:
    """Create an evidence-first Markdown report from finalized insight objects."""
    lines = [
        "# Customer Feedback Business Insights",
        "",
        "## Executive Summary",
        "",
        result.executive_summary,
        "",
        "## Key Findings",
        "",
    ]

    for _, row in result.findings.iterrows():
        lines.extend(
            [
                f'### {row["title"]}',
                "",
                f'**Evidence:** {row["evidence"]}',
                "",
                f'**Business interpretation:** {row["business_interpretation"]}',
                "",
            ]
        )
        review_ids = row.get("representative_review_ids")
        if isinstance(review_ids, (list, tuple)) and review_ids:
            lines.extend([f'**Supporting review IDs:** {", ".join(map(str, review_ids))}', ""])

    lines.extend(["## Recommendations", ""])
    if result.recommendations.empty:
        lines.extend(["No recommendation met the current evidence rules.", ""])
    else:
        for _, row in result.recommendations.iterrows():
            lines.extend(
                [
                    f'### [{row["priority"]}] {row["title"]}',
                    "",
                    row["explanation"],
                    "",
                    f'**Supporting metric:** {row["supporting_metric"]} = {float(row["supporting_value"]):.1%}',
                    f'**Supporting count:** {int(row["supporting_count"]):,}',
                    "",
                ]
            )

    lines.extend(["## Limitations", ""])
    lines.append(
        "- Recommendations are deterministic interpretations of observed feedback and do not establish causal relationships."
    )
    lines.append(
        "- Phase 6 reuses review-level sentiment for every detected aspect, so mixed-aspect reviews can be imperfectly represented."
    )
    if not result.trends.available:
        lines.append(f"- Trend analysis unavailable: {result.trends.reason}")
    return "\n".join(lines).strip() + "\n"
