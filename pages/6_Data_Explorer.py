"""Filter, inspect, and export the most enriched saved review dataset."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.dashboard.components import (
    dataframe_to_csv_bytes,
    render_empty_filtered_state,
    render_filter_status,
    render_global_filters,
    render_prerequisite,
)
from src.dashboard.errors import check_page_prerequisites
from src.dashboard.filters import apply_dashboard_filters
from src.dashboard.state import current_results_dataframe, initialize_session_state

APP_TITLE = "Data Explorer"


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🗂️", layout="wide")
    initialize_session_state(st.session_state)
    st.title(APP_TITLE)
    st.caption(
        "Explore saved canonical and enriched results. Filters update the view only; "
        "they never rerun sentiment, topic modeling, or aspect extraction."
    )

    status = check_page_prerequisites(st.session_state, "data_explorer")
    if not render_prerequisite(status):
        return

    dataframe = current_results_dataframe(st.session_state)
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        st.warning("No review data is available to explore.")
        return

    filters = render_global_filters(
        dataframe,
        st.session_state,
        key_prefix="explorer_global",
    )
    filtered = apply_dashboard_filters(dataframe, filters)
    st.session_state["filtered_df"] = filtered
    render_filter_status(len(dataframe), len(filtered), filters)

    if filtered.empty:
        render_empty_filtered_state()
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Reviews Analyzed", f"{len(filtered):,}")
    metric_cols[1].metric("Columns", len(filtered.columns))
    if "sentiment_label" in filtered.columns:
        metric_cols[2].metric(
            "Negative share", f"{filtered['sentiment_label'].astype(str).eq('Negative').mean():.1%}"
        )
    else:
        metric_cols[2].metric("Sentiment", "Not run")
    if "topic_label" in filtered.columns:
        metric_cols[3].metric("Topics visible", filtered["topic_label"].nunique())
    else:
        metric_cols[3].metric("Topics", "Not run")

    st.subheader("Review table")
    default_columns = [
        column
        for column in (
            "review_id", "review_text", "sentiment_label", "sentiment_score",
            "topic_label", "detected_aspects", "rating", "product", "category", "date",
        )
        if column in filtered.columns
    ]
    selected_columns = st.multiselect(
        "Columns to display",
        options=list(filtered.columns),
        default=default_columns or list(filtered.columns[: min(8, len(filtered.columns))]),
    )
    if not selected_columns:
        st.info("Select at least one column to preview records.")
    else:
        st.dataframe(
            filtered[selected_columns].head(500),
            use_container_width=True,
            hide_index=True,
        )
        if len(filtered) > 500:
            st.caption("Preview limited to 500 rows; downloads include every filtered row.")

    st.subheader("Export")
    st.download_button(
        "Download filtered enriched CSV",
        data=dataframe_to_csv_bytes(filtered),
        file_name="customer_feedback_filtered_enriched.csv",
        mime="text/csv",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
