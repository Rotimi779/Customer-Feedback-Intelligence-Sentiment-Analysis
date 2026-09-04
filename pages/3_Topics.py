"""Discover and explore Phase 5 NMF topics."""

from __future__ import annotations

import logging
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from src.pipeline import run_topic_stage
from src.topics.utils import TopicModelConfig, TopicModelError
from src.topics.visualization import (
    build_topic_distribution_chart,
    build_topic_frequency_chart,
    build_topic_keyword_table,
    build_topic_sentiment_chart,
)

APP_TITLE = "Topic Modeling"
LOGGER = logging.getLogger(__name__)


def _current_topic_signature() -> str:
    source = str(st.session_state.get("source_signature") or "")
    model = str(st.session_state.get("selected_sentiment_model") or "")
    return f"{source}:{model}"


def _render_quality(metrics: dict[str, object] | None) -> None:
    if not metrics:
        return
    st.subheader("Topic quality")
    columns = st.columns(4)
    columns[0].metric("Coverage", f"{float(metrics['topic_coverage']):.1%}")
    columns[1].metric("NPMI coherence", f"{float(metrics['topic_coherence_npmi']):.3f}")
    columns[2].metric("Topic diversity", f"{float(metrics['topic_diversity']):.3f}")
    stability = metrics.get("topic_stability")
    columns[3].metric(
        "Stability",
        "Not run" if stability is None else f"{float(stability):.3f}",
    )
    st.caption(
        "Coherence, diversity, coverage, and repeat-run stability are quantitative "
        "aids. Topic labels and representative reviews still require human "
        "interpretability review."
    )


def _render_summary(summary: pd.DataFrame, results_df: pd.DataFrame) -> None:
    if summary.empty:
        st.warning("No meaningful topic summary is available.")
        return

    largest = summary.iloc[0]
    smallest = summary.sort_values("review_count", ascending=True).iloc[0]
    metrics = st.columns(4)
    metrics[0].metric("Topics", f"{len(summary):,}")
    metrics[1].metric("Largest topic", str(largest["topic_label"]))
    metrics[2].metric("Smallest topic", str(smallest["topic_label"]))
    metrics[3].metric("Reviews assigned", f"{len(results_df):,}")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(build_topic_frequency_chart(summary), use_container_width=True)
    with right:
        st.plotly_chart(build_topic_distribution_chart(summary), use_container_width=True)

    sentiment_figure = build_topic_sentiment_chart(results_df)
    if sentiment_figure is not None:
        st.plotly_chart(sentiment_figure, use_container_width=True)

    st.subheader("Topic keywords")
    keyword_table = build_topic_keyword_table(summary)
    display = keyword_table.copy()
    display["percentage"] = display["percentage"].map(lambda value: f"{float(value):.1%}")
    if "average_sentiment" in display.columns:
        display["average_sentiment"] = display["average_sentiment"].map(
            lambda value: None if pd.isna(value) else round(float(value), 3)
        )
    st.dataframe(display, use_container_width=True, hide_index=True)


def _filter_topic_reviews(results_df: pd.DataFrame) -> pd.DataFrame:
    filtered = results_df.copy()
    with st.sidebar:
        st.header("Topic filters")
        topics = (
            filtered[["topic_id", "topic_label"]]
            .drop_duplicates()
            .sort_values("topic_id")
        )
        topic_options = ["All topics", *topics["topic_label"].astype(str).tolist()]
        selected_topic = st.selectbox("Topic", topic_options)
        if selected_topic != "All topics":
            filtered = filtered.loc[filtered["topic_label"].eq(selected_topic)]

        if "sentiment_label" in filtered.columns:
            labels = sorted(filtered["sentiment_label"].dropna().astype(str).unique())
            selected_sentiment = st.multiselect("Sentiment", labels, default=labels)
            filtered = (
                filtered.loc[filtered["sentiment_label"].isin(selected_sentiment)]
                if selected_sentiment
                else filtered.iloc[0:0]
            )

        keyword = st.text_input("Review keyword", placeholder="Search review text")
        if keyword.strip():
            filtered = filtered.loc[
                filtered["review_text"].astype(str).str.contains(
                    keyword.strip(), case=False, regex=False, na=False
                )
            ]

    return filtered.copy()


def _render_topic_explorer(
    summary: pd.DataFrame,
    results_df: pd.DataFrame,
    representatives: dict[int, list[str]] | None,
) -> None:
    st.subheader("Topic explorer")
    ordered = summary.sort_values("topic_id")
    labels = ordered["topic_label"].astype(str).tolist()
    selected_label = st.selectbox("Explore a topic", labels)
    selected_row = ordered.loc[ordered["topic_label"].eq(selected_label)].iloc[0]
    topic_id = int(selected_row["topic_id"])
    subset = results_df.loc[results_df["topic_id"].eq(topic_id)].copy()

    columns = st.columns(3)
    columns[0].metric("Topic size", f"{len(subset):,}")
    columns[1].metric("Dataset share", f"{float(selected_row['percentage']):.1%}")
    dominant = selected_row.get("dominant_sentiment")
    columns[2].metric(
        "Dominant sentiment",
        "Unavailable" if pd.isna(dominant) else str(dominant),
    )
    st.caption(f"Top keywords: {selected_row['top_keywords']}")

    if "sentiment_label" in subset.columns:
        sentiment_counts = (
            subset["sentiment_label"].value_counts().rename_axis("sentiment").reset_index(name="reviews")
        )
        figure = px.bar(
            sentiment_counts,
            x="sentiment",
            y="reviews",
            title=f"Sentiment within {selected_label}",
            labels={"sentiment": "Sentiment", "reviews": "Reviews"},
        )
        st.plotly_chart(figure, use_container_width=True)

    representative_subset = pd.DataFrame()
    if representatives and topic_id in representatives and "review_id" in subset.columns:
        wanted = representatives[topic_id]
        rank = {review_id: position for position, review_id in enumerate(wanted)}
        representative_subset = subset.loc[subset["review_id"].astype(str).isin(wanted)].copy()
        representative_subset["_rank"] = representative_subset["review_id"].astype(str).map(rank)
        representative_subset = representative_subset.sort_values("_rank").drop(columns="_rank")
    if representative_subset.empty:
        representative_subset = subset.head(5)

    st.markdown("**Representative reviews**")
    columns_to_show = ["review_text", "sentiment_label", "sentiment_score"]
    for optional in ("product", "date", "rating"):
        if optional in representative_subset.columns:
            columns_to_show.append(optional)
    st.dataframe(
        representative_subset[columns_to_show],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    """Render dataset-specific NMF topic modeling and exploration."""
    st.set_page_config(page_title=APP_TITLE, page_icon="🧩", layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "Discover dataset-specific themes with TF-IDF + Non-negative Matrix "
        "Factorization (NMF). The MVP intentionally uses one topic-modeling method."
    )

    results_df = st.session_state.get("results_df")
    if not isinstance(results_df, pd.DataFrame) or results_df.empty:
        st.warning("Run sentiment analysis before topic modeling.")
        st.page_link("pages/2_Sentiment.py", label="Go to Sentiment", icon="🙂")
        return

    required = {"sentiment_label", "sentiment_score"}
    if not required.issubset(results_df.columns):
        st.warning("The current results do not contain sentiment predictions yet.")
        st.page_link("pages/2_Sentiment.py", label="Go to Sentiment", icon="🙂")
        return

    unique_reviews = results_df["clean_text"].astype(str).str.strip().replace("", pd.NA).dropna().nunique()
    if unique_reviews < 5:
        st.warning("At least five usable unique reviews are needed for the MVP topic configuration.")
        return

    max_topics = min(15, int(unique_reviews))
    default_topics = min(8, max_topics)
    n_topics = st.slider(
        "Number of topics",
        min_value=5,
        max_value=max_topics,
        value=default_topics,
        help="The phase specification recommends experimenting with 5–15 topics.",
    )

    config = TopicModelConfig(n_topics=n_topics)
    run_clicked = st.button("Run Topic Modeling", type="primary", use_container_width=True)

    if run_clicked:
        try:
            with st.spinner("Discovering topics and evaluating stability..."):
                start = time.perf_counter()
                result = run_topic_stage(results_df, config=config, stability_runs=3)
                runtime = time.perf_counter() - start
        except (TopicModelError, ValueError) as exc:
            st.error(str(exc))
        except Exception:
            LOGGER.exception("Topic modeling failed")
            st.error(
                "Topic modeling could not be completed. Try fewer topics or confirm "
                "that the prepared reviews contain enough meaningful vocabulary."
            )
        else:
            st.session_state["results_df"] = result.dataframe
            st.session_state["topic_summary"] = result.summary
            st.session_state["topic_metrics"] = result.model.training_metadata.get("evaluation")
            st.session_state["topic_complete"] = True
            st.session_state["topic_source_signature"] = _current_topic_signature()
            st.session_state["topic_config"] = config.as_dict()
            st.session_state["topic_model_runtime"] = runtime
            st.session_state["topic_representatives"] = result.representative_review_ids
            # Topic changes invalidate downstream aspect/insight outputs.
            st.session_state["aspect_summary"] = None
            st.session_state["aspect_mentions"] = None
            st.session_state["aspect_metrics"] = None
            st.session_state["aspect_complete"] = False
            st.session_state["aspect_source_signature"] = None
            st.session_state["aspect_runtime_seconds"] = None
            st.session_state["insights"] = None
            st.session_state["insight_complete"] = False
            st.session_state["insight_source_signature"] = None
            st.session_state["insight_runtime_seconds"] = None

    summary = st.session_state.get("topic_summary")
    metrics = st.session_state.get("topic_metrics")
    stored_signature = st.session_state.get("topic_source_signature")
    if (
        isinstance(summary, pd.DataFrame)
        and not summary.empty
        and stored_signature == _current_topic_signature()
        and {"topic_id", "topic_label"}.issubset(st.session_state["results_df"].columns)
    ):
        current_results = st.session_state["results_df"]
        st.divider()
        runtime = st.session_state.get("topic_model_runtime")
        if runtime is not None:
            st.caption(f"Last topic-modeling runtime: {float(runtime):.3f} seconds.")
        _render_summary(summary, current_results)
        _render_quality(metrics)
        _render_topic_explorer(
            summary,
            current_results,
            st.session_state.get("topic_representatives"),
        )

        st.subheader("Filtered topic assignments")
        filtered = _filter_topic_reviews(current_results)
        if filtered.empty:
            st.info("No reviews match the active topic filters.")
        else:
            display_columns = [
                "review_id",
                "review_text",
                "sentiment_label",
                "topic_id",
                "topic_label",
            ]
            st.dataframe(
                filtered[[column for column in display_columns if column in filtered.columns]].head(200),
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()
