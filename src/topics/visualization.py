"""Reusable Plotly visualizations for topic-modeling results."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def build_topic_frequency_chart(summary: pd.DataFrame) -> go.Figure:
    """Build a ranked bar chart of review count by topic."""
    ordered = summary.sort_values("review_count", ascending=True)
    return px.bar(
        ordered,
        x="review_count",
        y="topic_label",
        orientation="h",
        title="Topic frequency",
        labels={"review_count": "Reviews", "topic_label": "Topic"},
    )


def build_topic_distribution_chart(summary: pd.DataFrame) -> go.Figure:
    """Build the phase-specified part-to-whole topic distribution chart."""
    return px.pie(
        summary,
        names="topic_label",
        values="review_count",
        title="Topic distribution",
    )


def build_topic_sentiment_chart(dataframe: pd.DataFrame) -> go.Figure | None:
    """Compare sentiment composition across discovered topics when available."""
    if "sentiment_label" not in dataframe.columns:
        return None
    counts = (
        dataframe.groupby(["topic_label", "sentiment_label"], observed=True)
        .size()
        .reset_index(name="reviews")
    )
    return px.bar(
        counts,
        x="topic_label",
        y="reviews",
        color="sentiment_label",
        barmode="stack",
        title="Sentiment within topics",
        labels={
            "topic_label": "Topic",
            "sentiment_label": "Sentiment",
            "reviews": "Reviews",
        },
    )


def build_topic_keyword_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the concise keyword table displayed by the Topics page."""
    columns = [
        "topic_id",
        "topic_label",
        "top_keywords",
        "review_count",
        "percentage",
    ]
    optional = [column for column in ("average_sentiment", "dominant_sentiment") if column in summary.columns]
    return summary[[*columns, *optional]].copy()
