"""Discover and explore Phase 5 NMF topics."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard.components import (
    render_empty_filtered_state,
    render_filter_status,
    render_global_filters,
    render_prerequisite,
)
from src.dashboard.errors import check_page_prerequisites
from src.dashboard.filters import apply_dashboard_filters
from src.dashboard.formatting import SENTIMENT_COLORS
from src.dashboard.state import initialize_session_state

from src.topics.visualization import (
    build_topic_distribution_chart,
    build_topic_frequency_chart,
    build_topic_keyword_table,
    build_topic_sentiment_chart,
)

APP_TITLE = "Topic Modeling"

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


def _filtered_topic_summary(base_summary: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """Reaggregate saved topic assignments for the active global-filter subset."""
    if results_df.empty:
        return base_summary.iloc[0:0].copy()
    total = len(results_df)
    rows: list[dict[str, object]] = []
    info = base_summary.set_index("topic_id", drop=False) if not base_summary.empty else pd.DataFrame()
    for topic_id, subset in results_df.groupby("topic_id", observed=True, sort=True):
        topic_id = int(topic_id)
        label = str(subset["topic_label"].iloc[0])
        original = info.loc[topic_id] if not info.empty and topic_id in info.index else None
        sentiment = subset["sentiment_label"].astype(str) if "sentiment_label" in subset.columns else pd.Series(dtype=str)
        score_map = {"Negative": -1.0, "Neutral": 0.0, "Positive": 1.0}
        numeric = sentiment.map(score_map).dropna()
        rows.append({
            "topic_id": topic_id,
            "topic_label": label,
            "top_keywords": "" if original is None else str(original.get("top_keywords", "")),
            "review_count": int(len(subset)),
            "percentage": float(len(subset) / total) if total else 0.0,
            "average_sentiment": float(numeric.mean()) if not numeric.empty else None,
            "dominant_sentiment": str(sentiment.value_counts().idxmax()) if not sentiment.empty else None,
        })
    return pd.DataFrame(rows).sort_values(["review_count", "topic_id"], ascending=[False, True]).reset_index(drop=True)


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
            color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            category_orders={"sentiment": ["Positive", "Neutral", "Negative"]},
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
    initialize_session_state(st.session_state)
    st.title(APP_TITLE)
    st.caption(
        "Explore the saved NMF topics produced by Run Full Analysis on the Upload & Setup page."
    )

    status = check_page_prerequisites(st.session_state, "topics")
    if not render_prerequisite(status):
        return
    results_df = st.session_state.get("results_df")

    required = {"sentiment_label", "sentiment_score"}
    if not required.issubset(results_df.columns):
        st.warning("The current results do not contain sentiment predictions yet.")
        st.page_link("pages/2_Sentiment.py", label="Go to Sentiment", icon="🙂")
        return

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
        filters = render_global_filters(
            current_results, st.session_state, key_prefix="topics_global"
        )
        filtered = apply_dashboard_filters(current_results, filters)
        render_filter_status(len(current_results), len(filtered), filters)

        st.divider()
        runtime = st.session_state.get("topic_model_runtime")
        if runtime is not None:
            st.caption(f"Last topic-modeling runtime: {float(runtime):.3f} seconds.")
        if filtered.empty:
            render_empty_filtered_state()
            return

        display_summary = _filtered_topic_summary(summary, filtered)
        _render_summary(display_summary, filtered)
        _render_quality(metrics)
        _render_topic_explorer(
            display_summary,
            filtered,
            st.session_state.get("topic_representatives"),
        )

        st.subheader("Filtered topic assignments")
        display_columns = [
            "review_id", "review_text", "sentiment_label", "topic_id", "topic_label"
        ]
        st.dataframe(
            filtered[[column for column in display_columns if column in filtered.columns]].head(200),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
