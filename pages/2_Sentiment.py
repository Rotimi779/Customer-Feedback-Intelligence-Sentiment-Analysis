"""Run and explore Phase 4 sentiment analysis results."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.formatting import SENTIMENT_COLORS
from src.dashboard.components import (
    render_empty_filtered_state,
    render_filter_status,
    render_global_filters,
    render_prerequisite,
)
from src.dashboard.errors import check_page_prerequisites
from src.dashboard.filters import apply_dashboard_filters
from src.dashboard.state import initialize_session_state

from src.sentiment import (
    SentimentModelName,
    load_production_model_selection,
    load_saved_model_comparison,
)

APP_TITLE = "Sentiment Analysis"


def _render_model_comparison() -> None:
    """Display saved model metrics when training/evaluation has been completed."""
    comparison = load_saved_model_comparison()
    if comparison.empty:
        st.info(
            "No saved model-evaluation reports are available yet. Train and evaluate "
            "the two sentiment models to populate this comparison."
        )
        return

    display = comparison.copy()
    for column in ("Logistic Regression", "DistilBERT"):
        if column in display.columns:
            display[column] = display[column].apply(
                lambda value: None
                if pd.isna(value)
                else round(float(value), 4)
                if isinstance(value, (float, int))
                else value
            )
    st.dataframe(display, use_container_width=True, hide_index=True)

    selection = load_production_model_selection()
    if selection:
        selected = SentimentModelName(selection["model_name"])
        st.success(
            f"Production model selected: **{selected.display_name}**. "
            f"Rationale: {selection['rationale']}"
        )
    else:
        st.caption(
            "Production model selection is intentionally not guessed. Review both "
            "evaluation reports, then record the choice with the evaluation CLI."
        )


def _filter_results(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply shared dashboard filters plus the sentiment-specific confidence threshold."""
    filters = render_global_filters(
        dataframe,
        st.session_state,
        key_prefix="sentiment_global",
    )
    filtered = apply_dashboard_filters(dataframe, filters)
    with st.sidebar:
        st.subheader("Sentiment options")
        minimum_confidence = st.slider(
            "Minimum confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="sentiment_min_confidence",
        )
    filtered = filtered.loc[filtered["sentiment_score"] >= minimum_confidence].copy()
    render_filter_status(len(dataframe), len(filtered), filters)
    return filtered


def _render_distribution(dataframe: pd.DataFrame) -> None:
    counts = (
        dataframe["sentiment_label"]
        .value_counts()
        .reindex(["Negative", "Neutral", "Positive"], fill_value=0)
        .rename_axis("sentiment")
        .reset_index(name="reviews")
    )
    figure = px.bar(
        counts,
        x="sentiment",
        y="reviews",
        color="sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"sentiment": ["Positive", "Neutral", "Negative"]},
        title="Sentiment distribution",
        labels={"sentiment": "Sentiment", "reviews": "Reviews"},
    )
    st.plotly_chart(figure, use_container_width=True)


def _render_confidence(dataframe: pd.DataFrame) -> None:
    figure = px.histogram(
        dataframe,
        x="sentiment_score",
        nbins=20,
        title="Prediction confidence distribution",
        labels={"sentiment_score": "Confidence", "count": "Reviews"},
    )
    st.plotly_chart(figure, use_container_width=True)


def _render_sentiment_trend(dataframe: pd.DataFrame) -> None:
    if "date" not in dataframe.columns:
        return
    parsed = pd.to_datetime(dataframe["date"], errors="coerce")
    usable = dataframe.assign(_date=parsed).dropna(subset=["_date"]).copy()
    if usable.empty:
        return

    usable["period"] = usable["_date"].dt.to_period("M").dt.to_timestamp()
    trend = (
        usable.groupby(["period", "sentiment_label"], observed=True)
        .size()
        .reset_index(name="reviews")
    )
    figure = px.line(
        trend,
        x="period",
        y="reviews",
        color="sentiment_label",
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={
            "sentiment_label": ["Positive", "Neutral", "Negative"]
        },
        markers=True,
        title="Sentiment over time",
        labels={
            "period": "Month",
            "reviews": "Reviews",
            "sentiment_label": "Sentiment",
        },
    )
    st.plotly_chart(figure, use_container_width=True)


def _render_representative_reviews(dataframe: pd.DataFrame) -> None:
    st.subheader("Representative reviews")
    for label in ("Negative", "Neutral", "Positive"):
        subset = (
            dataframe.loc[dataframe["sentiment_label"].eq(label)]
            .sort_values("sentiment_score", ascending=False)
            .head(3)
        )
        if subset.empty:
            continue
        with st.expander(f"{label} examples ({len(subset)})", expanded=False):
            columns = ["review_text", "sentiment_score"]
            for optional in ("date", "rating", "product", "category"):
                if optional in subset.columns:
                    columns.append(optional)
            st.dataframe(subset[columns], use_container_width=True, hide_index=True)


def _render_results(results_df: pd.DataFrame, model_name: SentimentModelName) -> None:
    filtered = _filter_results(results_df)
    if filtered.empty:
        render_empty_filtered_state()
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("Model", model_name.display_name)
    metric_columns[1].metric("Reviews", f"{len(filtered):,}")
    metric_columns[2].metric(
        "Average confidence",
        f"{filtered['sentiment_score'].mean():.1%}",
    )
    negative_share = filtered["sentiment_label"].eq("Negative").mean()
    metric_columns[3].metric("Negative share", f"{negative_share:.1%}")

    left, right = st.columns(2)
    with left:
        _render_distribution(filtered)
    with right:
        _render_confidence(filtered)

    _render_sentiment_trend(filtered)
    _render_representative_reviews(filtered)

    st.subheader("Enriched sentiment results")
    st.dataframe(filtered.head(200), use_container_width=True, hide_index=True)
    st.download_button(
        "Download sentiment results CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="customer_feedback_with_sentiment.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    """Render the complete Phase 4 sentiment page."""
    st.set_page_config(page_title=APP_TITLE, page_icon="🙂", layout="wide")
    initialize_session_state(st.session_state)
    st.title(APP_TITLE)
    st.caption(
        "Explore the saved sentiment results produced by Run Full Analysis on the "
        "Upload & Setup page."
    )

    status = check_page_prerequisites(st.session_state, "sentiment")
    if not render_prerequisite(status):
        return
    clean_df = st.session_state.get("canonical_df")
    if not isinstance(clean_df, pd.DataFrame):
        clean_df = st.session_state.get("clean_df")
    if clean_df.empty:
        st.warning("The prepared dataset has no usable reviews.")
        return

    st.subheader("Model evaluation")
    _render_model_comparison()
    st.divider()

    results_df = st.session_state.get("results_df")
    stored_model = st.session_state.get("selected_sentiment_model")
    stored_signature = st.session_state.get("sentiment_source_signature")
    source_signature = st.session_state.get("source_signature")
    if (
        isinstance(results_df, pd.DataFrame)
        and not results_df.empty
        and stored_model
        and stored_signature == source_signature
    ):
        st.divider()
        runtime = st.session_state.get("sentiment_runtime_seconds")
        if runtime is not None:
            st.caption(f"Last inference runtime: {float(runtime):.3f} seconds.")
        _render_results(results_df, SentimentModelName(stored_model))


if __name__ == "__main__":
    main()
