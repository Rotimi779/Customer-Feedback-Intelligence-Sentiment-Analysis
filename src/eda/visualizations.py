"""Reusable Plotly chart builders for exploratory analysis."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.eda.quality import missing_value_summary
from src.eda.statistics import (
    add_review_length_columns,
    calculate_common_words,
    infer_time_frequency,
    summarize_reviews_over_time,
)


def _finalize(
    figure: go.Figure,
    *,
    x_title: str,
    y_title: str,
) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        xaxis_title=x_title,
        yaxis_title=y_title,
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="closest",
    )
    return figure


def build_review_length_histogram(
    dataframe: pd.DataFrame,
    *,
    text_column: str = "review_text",
    bins: int = 30,
) -> go.Figure | None:
    """Build a histogram of review word counts."""
    if dataframe.empty or text_column not in dataframe.columns:
        return None
    lengths = add_review_length_columns(
        dataframe,
        text_column=text_column,
    )
    figure = px.histogram(
        lengths,
        x="review_word_count",
        nbins=bins,
        title="Review length distribution",
    )
    return _finalize(
        figure,
        x_title="Review length (words)",
        y_title="Number of reviews",
    )


def build_common_words_chart(
    dataframe: pd.DataFrame,
    *,
    text_column: str = "clean_text",
    top_n: int = 20,
) -> go.Figure | None:
    """Build a horizontal bar chart of common lightly processed words."""
    terms = calculate_common_words(
        dataframe,
        text_column=text_column,
        top_n=top_n,
    )
    if terms.empty:
        return None
    ordered = terms.sort_values("count", ascending=True)
    figure = px.bar(
        ordered,
        x="count",
        y="term",
        orientation="h",
        title=f"Top {len(ordered)} most common words",
        hover_data={"percentage": ":.1f"},
    )
    return _finalize(figure, x_title="Occurrences", y_title="Word")


def build_missing_values_chart(dataframe: pd.DataFrame) -> go.Figure | None:
    """Build a per-column missing-value percentage chart."""
    summary = missing_value_summary(dataframe)
    if summary.empty:
        return None
    ordered = summary.sort_values(
        ["missing_percentage", "column"],
        ascending=[True, True],
    )
    figure = px.bar(
        ordered,
        x="missing_percentage",
        y="column",
        orientation="h",
        title="Missing values by column",
        hover_data={"missing_count": True, "missing_percentage": ":.1f"},
    )
    figure.update_xaxes(range=[0, max(100.0, float(ordered["missing_percentage"].max()))])
    return _finalize(figure, x_title="Missing values (%)", y_title="Column")


def build_reviews_over_time_chart(
    dataframe: pd.DataFrame,
    *,
    date_column: str = "date",
) -> go.Figure | None:
    """Build a timeline only when usable date metadata exists."""
    frequency = infer_time_frequency(dataframe, date_column=date_column)
    timeline = summarize_reviews_over_time(
        dataframe,
        date_column=date_column,
        frequency=frequency,
    )
    if timeline.empty:
        return None
    labels = {"D": "day", "W": "week", "M": "month"}
    figure = px.line(
        timeline,
        x="period",
        y="review_count",
        markers=True,
        title=f"Reviews over time ({labels.get(frequency, frequency)})",
    )
    return _finalize(figure, x_title="Date", y_title="Number of reviews")


def build_rating_distribution_chart(
    dataframe: pd.DataFrame,
    *,
    rating_column: str = "rating",
) -> go.Figure | None:
    """Build a rating distribution when numeric ratings are available."""
    if rating_column not in dataframe.columns:
        return None
    ratings = pd.to_numeric(dataframe[rating_column], errors="coerce").dropna()
    if ratings.empty:
        return None
    rating_frame = pd.DataFrame({rating_column: ratings})
    figure = px.histogram(
        rating_frame,
        x=rating_column,
        nbins=min(20, max(5, int(ratings.nunique()))),
        title="Rating distribution",
    )
    return _finalize(
        figure,
        x_title="Rating",
        y_title="Number of reviews",
    )


def build_group_distribution_chart(
    dataframe: pd.DataFrame,
    *,
    column: str,
    top_n: int = 15,
) -> go.Figure | None:
    """Build a frequency chart for optional product or category metadata."""
    if column not in dataframe.columns:
        return None
    values = dataframe[column].astype("string").str.strip().replace("", pd.NA).dropna()
    if values.empty:
        return None
    counts = values.value_counts().head(top_n).sort_values(ascending=True)
    chart_data = counts.rename_axis(column).reset_index(name="review_count")
    title = column.replace("_", " ").title()
    figure = px.bar(
        chart_data,
        x="review_count",
        y=column,
        orientation="h",
        title=f"Reviews by {title.lower()}",
    )
    return _finalize(
        figure,
        x_title="Number of reviews",
        y_title=title,
    )
