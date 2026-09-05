"""Reusable Plotly visualizations for aspect-analysis outputs."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.dashboard.formatting import SENTIMENT_COLORS




def build_aspect_frequency_chart(summary: pd.DataFrame) -> go.Figure | None:
    """Rank aspects by mention count."""
    if summary.empty:
        return None
    ordered = summary.sort_values("mention_count", ascending=True)
    return px.bar(
        ordered,
        x="mention_count",
        y="aspect",
        orientation="h",
        title="Most discussed aspects",
        labels={"mention_count": "Aspect mentions", "aspect": "Aspect"},
    )


def build_aspect_sentiment_chart(mentions: pd.DataFrame) -> go.Figure | None:
    """Show positive, neutral, and negative mentions for each aspect."""
    if mentions.empty:
        return None
    counts = (
        mentions.groupby(["aspect", "aspect_sentiment_label"], observed=True)
        .size()
        .reset_index(name="mentions")
    )
    return px.bar(
        counts,
        x="aspect",
        y="mentions",
        color="aspect_sentiment_label",
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={
            "aspect_sentiment_label": ["Positive", "Neutral", "Negative"]
        },
        barmode="stack",
        title="Aspect sentiment comparison",
        labels={
            "aspect": "Aspect",
            "aspect_sentiment_label": "Sentiment",
            "mentions": "Mentions",
        },
    )


def build_positive_negative_chart(summary: pd.DataFrame) -> go.Figure | None:
    """Compare positive and negative aspect mentions directly."""
    if summary.empty:
        return None
    long = summary[["aspect", "positive_count", "negative_count"]].melt(
        id_vars="aspect",
        var_name="sentiment",
        value_name="mentions",
    )
    long["sentiment"] = long["sentiment"].map(
        {"positive_count": "Positive", "negative_count": "Negative"}
    )
    return px.bar(
        long,
        x="aspect",
        y="mentions",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"sentiment": ["Positive", "Negative"]},
        barmode="group",
        title="Positive vs negative aspect mentions",
        labels={"aspect": "Aspect", "mentions": "Mentions", "sentiment": "Sentiment"},
    )


def build_aspect_rating_chart(summary: pd.DataFrame) -> go.Figure | None:
    """Rank average rating by aspect when rating metadata is available."""
    if summary.empty or "average_rating" not in summary.columns:
        return None
    usable = summary.dropna(subset=["average_rating"]).copy()
    if usable.empty:
        return None
    usable = usable.sort_values("average_rating", ascending=True)
    return px.bar(
        usable,
        x="average_rating",
        y="aspect",
        orientation="h",
        title="Average rating by aspect",
        labels={"average_rating": "Average rating", "aspect": "Aspect"},
    )
