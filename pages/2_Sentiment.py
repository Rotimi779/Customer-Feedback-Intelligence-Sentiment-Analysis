"""Run and explore Phase 4 sentiment analysis results."""

from __future__ import annotations

import logging
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from src.sentiment import (
    SentimentAnalyzer,
    SentimentModelName,
    available_sentiment_models,
    load_production_model_selection,
    load_saved_model_comparison,
)

APP_TITLE = "Sentiment Analysis"
LOGGER = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def _load_analyzer(model_name: str) -> SentimentAnalyzer:
    """Cache heavy model artifacts across ordinary Streamlit reruns."""
    return SentimentAnalyzer.load(SentimentModelName(model_name))


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
    """Apply sentiment-page filters without mutating stored results."""
    filtered = dataframe.copy()
    with st.sidebar:
        st.header("Sentiment filters")
        labels = sorted(filtered["sentiment_label"].dropna().astype(str).unique())
        selected_labels = st.multiselect(
            "Sentiment",
            options=labels,
            default=labels,
        )
        if selected_labels:
            filtered = filtered.loc[filtered["sentiment_label"].isin(selected_labels)]
        else:
            filtered = filtered.iloc[0:0]

        minimum_confidence = st.slider(
            "Minimum confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
        )
        filtered = filtered.loc[filtered["sentiment_score"] >= minimum_confidence]

        if "product" in filtered.columns:
            products = sorted(filtered["product"].dropna().astype(str).unique())
            selected_products = st.multiselect("Products", options=products)
            if selected_products:
                filtered = filtered.loc[
                    filtered["product"].astype(str).isin(selected_products)
                ]

        keyword = st.text_input("Keyword search", placeholder="Search review text")
        if keyword.strip():
            filtered = filtered.loc[
                filtered["review_text"].astype(str).str.contains(
                    keyword.strip(),
                    case=False,
                    regex=False,
                    na=False,
                )
            ]

    return filtered.copy()


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
        st.warning("No sentiment results match the active filters.")
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
    st.title(APP_TITLE)
    st.caption(
        "Compare the classical baseline with DistilBERT, then run the selected "
        "local model against the prepared canonical dataset."
    )

    clean_df = st.session_state.get("clean_df")
    if clean_df is None or not isinstance(clean_df, pd.DataFrame):
        st.warning("Prepare a dataset on the Upload & Setup page first.")
        st.page_link("app.py", label="Go to Upload & Setup", icon="📤")
        return
    if clean_df.empty:
        st.warning("The prepared dataset has no usable reviews.")
        return

    st.subheader("Model evaluation")
    _render_model_comparison()
    st.divider()

    available = available_sentiment_models()
    if not available:
        st.warning(
            "No trained sentiment artifacts were found. The Phase 4 code is ready, "
            "but inference requires training at least one local model first."
        )
        st.code(
            "python -m src.sentiment.train_baseline --input data/training/YOUR_LABELLED_DATA.csv\n"
            "python -m src.sentiment.train_transformer --input data/training/YOUR_LABELLED_DATA.csv",
            language="powershell",
        )
        st.caption(
            "Use the same labelled dataset and split seed for a fair comparison. "
            "The training CSV must contain review_text and three-class sentiment_label columns."
        )
        return

    selection = load_production_model_selection()
    preferred = None
    if selection:
        try:
            candidate = SentimentModelName(selection["model_name"])
            if candidate in available:
                preferred = candidate
        except (KeyError, ValueError):
            preferred = None
    preferred = preferred or available[0]

    selected_model = st.selectbox(
        "Inference model",
        options=available,
        index=available.index(preferred),
        format_func=lambda model: model.display_name,
        help="Only locally trained model artifacts are shown here.",
    )

    progress = st.progress(0.0, text="Ready to run sentiment inference.")
    run_clicked = st.button(
        "Run Sentiment Analysis",
        type="primary",
        use_container_width=True,
    )

    if run_clicked:
        try:
            analyzer = _load_analyzer(selected_model.value)
            start = time.perf_counter()

            def update_progress(done: int, total: int) -> None:
                fraction = 1.0 if total == 0 else done / total
                progress.progress(
                    fraction,
                    text=f"Classifying reviews: {done:,} / {total:,}",
                )

            result = analyzer.predict_dataframe(
                clean_df,
                progress_callback=update_progress,
            )
            runtime_seconds = time.perf_counter() - start
        except Exception:
            LOGGER.exception("Sentiment inference failed for model=%s", selected_model.value)
            st.error(
                "Sentiment inference could not be completed. Confirm that the selected "
                "model artifacts are present and compatible with this project."
            )
        else:
            progress.progress(1.0, text="Sentiment inference complete.")
            st.session_state["results_df"] = result.dataframe
            st.session_state["sentiment_complete"] = True
            st.session_state["selected_sentiment_model"] = selected_model.value
            st.session_state["sentiment_runtime_seconds"] = runtime_seconds
            st.session_state["sentiment_source_signature"] = st.session_state.get(
                "source_signature"
            )
            # A new sentiment result invalidates all downstream topic outputs.
            st.session_state["topic_complete"] = False
            st.session_state["topic_summary"] = None
            st.session_state["topic_metrics"] = None
            st.session_state["topic_source_signature"] = None
            st.session_state["topic_config"] = None
            st.session_state["topic_model_runtime"] = None
            st.session_state["topic_representatives"] = None

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
