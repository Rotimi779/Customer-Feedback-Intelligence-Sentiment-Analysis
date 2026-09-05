"""Generate and explore Phase 7 evidence-backed business insights."""

from __future__ import annotations

import hashlib
import logging
import time

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

from src.insights import (
    BusinessInsightsResult,
    InsightsError,
    build_markdown_report,
    dataframe_to_csv_bytes,
    recommendations_to_csv_bytes,
)
from src.pipeline import run_insight_stage

APP_TITLE = "Business Insights"
LOGGER = logging.getLogger(__name__)


def _current_insight_signature() -> str:
    """Tie generated insights to the exact upstream Phase 6 result."""
    parts = [
        str(st.session_state.get("source_signature") or ""),
        str(st.session_state.get("selected_sentiment_model") or ""),
        str(st.session_state.get("topic_source_signature") or ""),
        str(st.session_state.get("topic_config") or ""),
        str(st.session_state.get("aspect_source_signature") or ""),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _render_kpis(result: BusinessInsightsResult) -> None:
    """Show the minimum decision-oriented metrics required by Phase 7."""
    metrics = result.metrics
    sentiment = metrics["sentiment"]
    topic = metrics.get("most_discussed_topic")
    pain = metrics.get("priority_improvement")
    strength = metrics.get("key_strength")

    cards = st.columns(5)
    cards[0].metric("Reviews analyzed", f"{int(metrics['total_reviews']):,}")
    cards[1].metric(
        "Overall sentiment",
        str(sentiment.get("dominant_sentiment") or "Unavailable"),
        f"{float(sentiment.get(str(sentiment.get('dominant_sentiment', '')).lower() + '_share', 0.0)):.1%}",
    )
    cards[2].metric(
        "Top topic",
        str(topic["topic_label"]) if topic else "Unavailable",
        f"{float(topic['share']):.1%}" if topic else None,
    )
    cards[3].metric(
        "Priority improvement",
        str(pain["aspect"]) if pain else "No supported aspect",
        f"{float(pain['negative_share']):.1%} negative" if pain else None,
    )
    cards[4].metric(
        "Key strength",
        str(strength["aspect"]) if strength else "No supported aspect",
        f"{float(strength['positive_share']):.1%} positive" if strength else None,
    )


def _render_findings(result: BusinessInsightsResult) -> None:
    st.subheader("Key findings")
    if result.findings.empty:
        st.info("No evidence-backed findings are available for this result.")
        return

    for _, row in result.findings.iterrows():
        with st.container(border=True):
            st.markdown(f"**{row['title']}**")
            st.write(str(row["evidence"]))
            st.caption(str(row["business_interpretation"]))
            review_ids = row.get("representative_review_ids")
            if isinstance(review_ids, (list, tuple)) and review_ids:
                st.caption("Supporting review IDs: " + ", ".join(map(str, review_ids)))


def _render_recommendations(result: BusinessInsightsResult) -> None:
    st.subheader("Recommendations")
    st.caption(
        "Recommendations are deterministic and evidence-backed. They suggest areas to "
        "investigate; they do not claim that observed feedback establishes causation."
    )
    if result.recommendations.empty:
        st.info("No recommendation met the current evidence rules.")
        return

    display = result.recommendations.copy()
    display["supporting_value"] = display.apply(
        lambda row: f"{float(row['supporting_value']):.1%}", axis=1
    )
    if "representative_review_ids" in display.columns:
        display["representative_review_ids"] = display["representative_review_ids"].apply(
            lambda value: ", ".join(map(str, value))
            if isinstance(value, (list, tuple))
            else ""
        )
    st.dataframe(
        display[
            [
                "priority",
                "title",
                "affected_item",
                "supporting_metric",
                "supporting_value",
                "supporting_count",
                "explanation",
                "representative_review_ids",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def _render_trends(result: BusinessInsightsResult) -> None:
    st.subheader("Trends")
    trends = result.trends
    if not trends.available:
        st.info(trends.reason or "Trend analysis is unavailable for this dataset.")
        return

    label = {"D": "daily", "W": "weekly", "M": "monthly"}.get(
        str(trends.frequency), str(trends.frequency)
    )
    st.caption(f"Trend tables use {label} periods based on the mapped date span.")

    signal_cards = st.columns(3)
    if trends.fastest_growing_topic:
        item = trends.fastest_growing_topic
        signal_cards[0].metric(
            "Fastest-growing topic",
            str(item["topic_label"]),
            f"+{float(item['share_change']) * 100:.1f} pp",
        )
    else:
        signal_cards[0].metric("Fastest-growing topic", "No supported signal")

    if trends.worsening_aspect:
        item = trends.worsening_aspect
        signal_cards[1].metric(
            "Worsening aspect",
            str(item["aspect"]),
            f"+{float(item['negative_share_change']) * 100:.1f} pp negative",
            delta_color="inverse",
        )
    else:
        signal_cards[1].metric("Worsening aspect", "No supported signal")

    change = trends.sentiment_change
    if change:
        delta = float(change["negative_share_change"])
        signal_cards[2].metric(
            "Negative sentiment change",
            f"{float(change['recent_negative_share']):.1%}",
            f"{delta * 100:+.1f} pp",
            delta_color="inverse",
        )
    else:
        signal_cards[2].metric("Negative sentiment change", "Unavailable")

    left, right = st.columns(2)
    if not trends.review_volume.empty:
        volume_chart = px.line(
            trends.review_volume,
            x="period",
            y="review_count",
            markers=True,
            title="Review volume over time",
            labels={"period": "Period", "review_count": "Reviews"},
        )
        with left:
            st.plotly_chart(volume_chart, use_container_width=True)

    if not trends.sentiment.empty:
        sentiment_chart = px.line(
            trends.sentiment,
            x="period",
            y="share",
            color="sentiment_label",
            markers=True,
            color_discrete_map=SENTIMENT_COLORS,
            category_orders={"sentiment_label": ["Positive", "Neutral", "Negative"]},
            title="Sentiment share over time",
            labels={"period": "Period", "share": "Share", "sentiment_label": "Sentiment"},
        )
        sentiment_chart.update_yaxes(tickformat=".0%")
        with right:
            st.plotly_chart(sentiment_chart, use_container_width=True)


def _render_exports(result: BusinessInsightsResult) -> None:
    st.subheader("Download results")
    report = build_markdown_report(result)
    columns = st.columns(3)
    columns[0].download_button(
        "Download enriched CSV",
        data=dataframe_to_csv_bytes(result.dataframe),
        file_name="customer_feedback_enriched.csv",
        mime="text/csv",
        use_container_width=True,
    )
    columns[1].download_button(
        "Download insight report",
        data=report.encode("utf-8"),
        file_name="business_insights.md",
        mime="text/markdown",
        use_container_width=True,
    )
    columns[2].download_button(
        "Download recommendations",
        data=recommendations_to_csv_bytes(result.recommendations),
        file_name="recommendations.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    """Render the Phase 7 deterministic business-insight workflow."""
    st.set_page_config(page_title=APP_TITLE, page_icon="💡", layout="wide")
    initialize_session_state(st.session_state)
    st.title(APP_TITLE)
    st.caption(
        "Turn sentiment, topic, and aspect outputs into concise evidence-backed findings, "
        "cautious recommendations, and optional date-based trends."
    )

    status = check_page_prerequisites(st.session_state, "insights")
    if not render_prerequisite(status):
        return
    results_df = st.session_state.get("results_df")

    required = {
        "sentiment_label",
        "topic_id",
        "topic_label",
        "detected_aspects",
        "aspect_sentiment",
    }
    if not required.issubset(results_df.columns) or not st.session_state.get("aspect_complete"):
        st.warning("Phase 7 requires the current Phase 6 aspect-enriched analysis results.")
        st.page_link("pages/4_Aspect_Analysis.py", label="Go to Aspect Analysis", icon="🔎")
        return

    if st.button("Generate Business Insights", type="primary", use_container_width=True):
        try:
            with st.spinner("Aggregating evidence and generating deterministic insights..."):
                start = time.perf_counter()
                result = run_insight_stage(
                    results_df,
                    topic_summary=st.session_state.get("topic_summary"),
                    aspect_summary=st.session_state.get("aspect_summary"),
                    aspect_mentions=st.session_state.get("aspect_mentions"),
                )
                runtime = time.perf_counter() - start
        except (InsightsError, ValueError) as exc:
            st.error(str(exc))
        except Exception:
            LOGGER.exception("Business insight generation failed")
            st.error(
                "Business insights could not be generated. Confirm that sentiment, topic, "
                "and aspect outputs are current and try again."
            )
        else:
            st.session_state["insights"] = result
            st.session_state["insight_complete"] = True
            st.session_state["insight_source_signature"] = _current_insight_signature()
            st.session_state["insight_runtime_seconds"] = runtime
            st.session_state["analysis_complete"] = True

    result = st.session_state.get("insights")
    stored_signature = st.session_state.get("insight_source_signature")
    if (
        isinstance(result, BusinessInsightsResult)
        and stored_signature == _current_insight_signature()
    ):
        filters = render_global_filters(
            results_df, st.session_state, key_prefix="insights_global"
        )
        filtered_results = apply_dashboard_filters(results_df, filters)
        render_filter_status(len(results_df), len(filtered_results), filters)

        st.divider()
        runtime = st.session_state.get("insight_runtime_seconds")
        if runtime is not None:
            st.caption(f"Last insight-generation runtime: {float(runtime):.3f} seconds.")
        if filtered_results.empty:
            render_empty_filtered_state()
            return

        display_result = result
        if filters.is_active():
            # Recompute deterministic aggregates only; this does not rerun any ML model.
            try:
                display_result = run_insight_stage(filtered_results)
            except (InsightsError, ValueError) as exc:
                st.warning(f"Filtered insights are unavailable: {exc}")
                return

        st.subheader("Executive summary")
        st.write(display_result.executive_summary)
        _render_kpis(display_result)
        _render_findings(display_result)
        _render_recommendations(display_result)
        _render_trends(display_result)

        st.caption(
            "Known MVP limitation: aspect sentiment reuses review-level sentiment, so a "
            "single mixed review may not express the true sentiment of every detected aspect."
        )
        _render_exports(display_result)


if __name__ == "__main__":
    main()
