"""Run and explore Phase 6 rule-based aspect analysis."""

from __future__ import annotations

import hashlib
import logging
import time

import pandas as pd
import streamlit as st

from src.aspects import AspectAnalysisError
from src.aspects.visualization import (
    build_aspect_frequency_chart,
    build_aspect_rating_chart,
    build_aspect_sentiment_chart,
    build_positive_negative_chart,
)
from src.pipeline import run_aspect_stage

APP_TITLE = "Aspect Analysis"
LOGGER = logging.getLogger(__name__)


def _current_aspect_signature() -> str:
    """Tie stored aspect outputs to the current Phase 5 result."""
    parts = [
        str(st.session_state.get("source_signature") or ""),
        str(st.session_state.get("selected_sentiment_model") or ""),
        str(st.session_state.get("topic_source_signature") or ""),
        str(st.session_state.get("topic_config") or ""),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _render_quality(metrics: dict[str, object] | None) -> None:
    """Show measurable structural checks without pretending they are accuracy."""
    if not metrics:
        return

    st.subheader("Aspect extraction coverage")
    cards = st.columns(4)
    cards[0].metric("Reviews with aspects", f"{int(metrics['reviews_with_aspects']):,}")
    cards[1].metric("Coverage", f"{float(metrics['aspect_coverage']):.1%}")
    cards[2].metric("Aspect mentions", f"{int(metrics['total_aspect_mentions']):,}")
    cards[3].metric("Unique aspects", f"{int(metrics['unique_aspects']):,}")

    if bool(metrics.get("manual_review_required")):
        st.caption(str(metrics.get("manual_review_note", "Manual review is required.")))


def _render_summary(summary: pd.DataFrame, mentions: pd.DataFrame) -> None:
    """Render aggregate aspect metrics and required charts."""
    if summary.empty:
        st.info(
            "No aspects were detected with the current MVP vocabulary. The rule-based "
            "extractor only reports explicit configured keywords and phrases."
        )
        return

    most_discussed = summary.iloc[0]
    most_positive = summary.sort_values(
        ["positive_share", "mention_count"], ascending=[False, False]
    ).iloc[0]
    most_negative = summary.sort_values(
        ["negative_share", "mention_count"], ascending=[False, False]
    ).iloc[0]

    cards = st.columns(3)
    cards[0].metric("Most discussed", str(most_discussed["aspect"]))
    cards[1].metric("Most positive", str(most_positive["aspect"]))
    cards[2].metric("Most negative", str(most_negative["aspect"]))

    left, right = st.columns(2)
    frequency = build_aspect_frequency_chart(summary)
    comparison = build_positive_negative_chart(summary)
    if frequency is not None:
        with left:
            st.plotly_chart(frequency, use_container_width=True)
    if comparison is not None:
        with right:
            st.plotly_chart(comparison, use_container_width=True)

    sentiment = build_aspect_sentiment_chart(mentions)
    if sentiment is not None:
        st.plotly_chart(sentiment, use_container_width=True)

    rating = build_aspect_rating_chart(summary)
    if rating is not None:
        st.subheader("Rating by aspect")
        st.plotly_chart(rating, use_container_width=True)
        rated = summary.dropna(subset=["average_rating"])
        if not rated.empty:
            highest = rated.sort_values("average_rating", ascending=False).iloc[0]
            lowest = rated.sort_values("average_rating", ascending=True).iloc[0]
            rating_cards = st.columns(2)
            rating_cards[0].metric(
                "Highest-rated aspect",
                str(highest["aspect"]),
                f"{float(highest['average_rating']):.2f}",
            )
            rating_cards[1].metric(
                "Lowest-rated aspect",
                str(lowest["aspect"]),
                f"{float(lowest['average_rating']):.2f}",
            )
    else:
        st.caption("Rating-based aspect rankings are unavailable because no usable rating metadata exists.")

    st.subheader("Aspect summary")
    display = summary.copy()
    for column in ("review_coverage", "positive_share", "neutral_share", "negative_share"):
        display[column] = display[column].map(lambda value: f"{float(value):.1%}")
    for column in ("average_sentiment", "average_confidence", "average_rating"):
        display[column] = display[column].map(
            lambda value: None if pd.isna(value) else round(float(value), 3)
        )
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_aspect_explorer(
    summary: pd.DataFrame,
    mentions: pd.DataFrame,
) -> None:
    """Let a user inspect one aspect and its supporting reviews."""
    if summary.empty or mentions.empty:
        return

    st.subheader("Aspect explorer")
    aspects = summary["aspect"].astype(str).tolist()
    selected = st.selectbox("Inspect an aspect", aspects)
    row = summary.loc[summary["aspect"].eq(selected)].iloc[0]
    subset = mentions.loc[mentions["aspect"].eq(selected)].copy()

    cards = st.columns(4)
    cards[0].metric("Mentions", f"{int(row['mention_count']):,}")
    cards[1].metric("Positive", f"{float(row['positive_share']):.1%}")
    cards[2].metric("Neutral", f"{float(row['neutral_share']):.1%}")
    cards[3].metric("Negative", f"{float(row['negative_share']):.1%}")

    if "topic_label" in subset.columns:
        topic_counts = subset["topic_label"].dropna().astype(str).value_counts().head(5)
        if not topic_counts.empty:
            st.caption(
                "Associated topics: "
                + ", ".join(
                    f"{topic} ({count})" for topic, count in topic_counts.items()
                )
            )

    st.markdown("**Supporting reviews**")
    columns = [
        column
        for column in (
            "review_text",
            "aspect_sentiment_label",
            "aspect_confidence",
            "topic_label",
            "rating",
            "product",
            "date",
        )
        if column in subset.columns
    ]
    st.dataframe(subset[columns].head(100), use_container_width=True, hide_index=True)


def _filter_mentions(mentions: pd.DataFrame) -> pd.DataFrame:
    """Apply simple aspect/sentiment/topic filters to the evidence table."""
    filtered = mentions.copy()
    with st.sidebar:
        st.header("Aspect filters")
        aspects = sorted(filtered["aspect"].dropna().astype(str).unique())
        selected_aspects = st.multiselect("Aspects", aspects, default=aspects)
        filtered = (
            filtered.loc[filtered["aspect"].isin(selected_aspects)]
            if selected_aspects
            else filtered.iloc[0:0]
        )

        if "aspect_sentiment_label" in filtered.columns:
            labels = sorted(filtered["aspect_sentiment_label"].dropna().astype(str).unique())
            selected_labels = st.multiselect("Sentiment", labels, default=labels)
            filtered = (
                filtered.loc[filtered["aspect_sentiment_label"].isin(selected_labels)]
                if selected_labels
                else filtered.iloc[0:0]
            )

        if "topic_label" in filtered.columns:
            topics = sorted(filtered["topic_label"].dropna().astype(str).unique())
            selected_topics = st.multiselect("Topics", topics)
            if selected_topics:
                filtered = filtered.loc[filtered["topic_label"].isin(selected_topics)]

        keyword = st.text_input("Review keyword", placeholder="Search review text")
        if keyword.strip() and "review_text" in filtered.columns:
            filtered = filtered.loc[
                filtered["review_text"].astype(str).str.contains(
                    keyword.strip(), case=False, regex=False, na=False
                )
            ]
    return filtered.copy()


def main() -> None:
    """Render Phase 6 rule-based aspect extraction and aggregation."""
    st.set_page_config(page_title=APP_TITLE, page_icon="🔎", layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "Identify explicit product/service aspects using transparent keyword, synonym, "
        "and phrase rules, then reuse the Phase 4 review-level sentiment for each match."
    )

    results_df = st.session_state.get("results_df")
    if not isinstance(results_df, pd.DataFrame) or results_df.empty:
        st.warning("Run sentiment and topic modeling before Aspect Analysis.")
        st.page_link("pages/3_Topics.py", label="Go to Topics", icon="🧩")
        return

    required = {"sentiment_label", "sentiment_score", "topic_id", "topic_label"}
    if not required.issubset(results_df.columns):
        st.warning("Phase 6 requires the current Phase 5 topic-enriched sentiment results.")
        st.page_link("pages/3_Topics.py", label="Go to Topics", icon="🧩")
        return

    if st.button("Run Aspect Analysis", type="primary", use_container_width=True):
        try:
            with st.spinner("Extracting aspects and aggregating sentiment..."):
                start = time.perf_counter()
                result = run_aspect_stage(results_df)
                runtime = time.perf_counter() - start
        except (AspectAnalysisError, ValueError) as exc:
            st.error(str(exc))
        except Exception:
            LOGGER.exception("Aspect analysis failed")
            st.error(
                "Aspect analysis could not be completed. Confirm that sentiment and "
                "topic outputs are available and try again."
            )
        else:
            st.session_state["results_df"] = result.dataframe
            st.session_state["aspect_summary"] = result.summary
            st.session_state["aspect_mentions"] = result.mentions
            st.session_state["aspect_metrics"] = result.evaluation
            st.session_state["aspect_complete"] = True
            st.session_state["aspect_source_signature"] = _current_aspect_signature()
            st.session_state["aspect_runtime_seconds"] = runtime
            st.session_state["insights"] = None

    summary = st.session_state.get("aspect_summary")
    mentions = st.session_state.get("aspect_mentions")
    stored_signature = st.session_state.get("aspect_source_signature")
    current_results = st.session_state.get("results_df")

    if (
        isinstance(summary, pd.DataFrame)
        and isinstance(mentions, pd.DataFrame)
        and isinstance(current_results, pd.DataFrame)
        and stored_signature == _current_aspect_signature()
        and {"detected_aspects", "aspect_sentiment"}.issubset(current_results.columns)
    ):
        st.divider()
        runtime = st.session_state.get("aspect_runtime_seconds")
        if runtime is not None:
            st.caption(f"Last aspect-analysis runtime: {float(runtime):.3f} seconds.")

        _render_quality(st.session_state.get("aspect_metrics"))
        _render_summary(summary, mentions)
        _render_aspect_explorer(summary, mentions)

        st.subheader("Filtered aspect evidence")
        if mentions.empty:
            st.info("No aspect evidence is available for the current dataset.")
        else:
            filtered = _filter_mentions(mentions)
            if filtered.empty:
                st.info("No aspect mentions match the active filters.")
            else:
                display_columns = [
                    column
                    for column in (
                        "review_id",
                        "review_text",
                        "aspect",
                        "aspect_sentiment_label",
                        "aspect_confidence",
                        "topic_label",
                        "rating",
                        "product",
                        "date",
                    )
                    if column in filtered.columns
                ]
                st.dataframe(
                    filtered[display_columns].head(200),
                    use_container_width=True,
                    hide_index=True,
                )


if __name__ == "__main__":
    main()
