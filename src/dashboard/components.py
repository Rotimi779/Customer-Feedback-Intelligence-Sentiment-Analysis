"""Reusable Streamlit presentation helpers for the integrated dashboard."""

from __future__ import annotations

from typing import Any, MutableMapping

import pandas as pd
import streamlit as st

from src.dashboard.errors import PrerequisiteStatus
from src.dashboard.filters import DashboardFilters, get_dashboard_filter_options
from src.dashboard.formatting import format_percentage


def render_prerequisite(status: PrerequisiteStatus) -> bool:
    """Render one prerequisite warning and return whether the page may continue."""
    if status.ready:
        return True
    st.warning(status.message or "This page is not ready yet.")
    if status.next_page:
        label = "Go to required step"
        st.page_link(status.next_page, label=label, icon="➡️")
    return False


def render_analysis_status(state: MutableMapping[str, Any]) -> None:
    """Show a compact status banner for the current dataset and pipeline."""
    if state.get("analysis_running"):
        stage = str(state.get("analysis_stage") or "analysis").replace("_", " ").title()
        st.info(f"Analysis is running: {stage}.")
    elif state.get("analysis_complete"):
        st.success("Full analysis is complete. Results are available across all pages.")
    elif state.get("clean_df") is not None:
        st.info("Dataset confirmed. Run the full pipeline or execute stages from their pages.")

    error = state.get("last_error")
    if error:
        st.error(str(error))


def render_global_filters(
    dataframe: pd.DataFrame,
    state: MutableMapping[str, Any],
    *,
    key_prefix: str,
) -> DashboardFilters:
    """Render a consistent adaptive sidebar filter set and persist selections."""
    options = get_dashboard_filter_options(dataframe)
    saved = DashboardFilters.from_dict(state.get("active_filters"))

    with st.sidebar:
        st.header("Global filters")
        if st.button("Reset global filters", key=f"{key_prefix}_reset", use_container_width=True):
            state["active_filters"] = {}
            for widget_key in list(state.keys()):
                if str(widget_key).startswith(f"{key_prefix}_") and widget_key != f"{key_prefix}_reset":
                    state.pop(widget_key, None)
            saved = DashboardFilters()

        start_date = saved.start_date
        end_date = saved.end_date
        if options.minimum_date is not None and options.maximum_date is not None:
            min_date = options.minimum_date.date()
            max_date = options.maximum_date.date()
            start_default = start_date if start_date and min_date <= start_date <= max_date else min_date
            end_default = end_date if end_date and min_date <= end_date <= max_date else max_date
            date_range = st.date_input(
                "Date range",
                value=(start_default, end_default),
                min_value=min_date,
                max_value=max_date,
                key=f"{key_prefix}_date_range",
            )
            if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                start_date, end_date = date_range

        products = tuple(st.multiselect(
            "Product", options=options.products,
            default=[v for v in saved.products if v in options.products],
            key=f"{key_prefix}_products",
        )) if options.products else saved.products

        categories = tuple(st.multiselect(
            "Category", options=options.categories,
            default=[v for v in saved.categories if v in options.categories],
            key=f"{key_prefix}_categories",
        )) if options.categories else saved.categories

        rating_min, rating_max = saved.rating_min, saved.rating_max
        if options.minimum_rating is not None and options.maximum_rating is not None:
            if options.minimum_rating < options.maximum_rating:
                low = saved.rating_min if saved.rating_min is not None else options.minimum_rating
                high = saved.rating_max if saved.rating_max is not None else options.maximum_rating
                rating_min, rating_max = st.slider(
                    "Rating range",
                    min_value=float(options.minimum_rating),
                    max_value=float(options.maximum_rating),
                    value=(float(max(options.minimum_rating, low)), float(min(options.maximum_rating, high))),
                    key=f"{key_prefix}_ratings",
                )
            else:
                rating_min = rating_max = options.minimum_rating

        sentiments = tuple(st.multiselect(
            "Sentiment", options=options.sentiments,
            default=[v for v in saved.sentiments if v in options.sentiments],
            key=f"{key_prefix}_sentiments",
        )) if options.sentiments else saved.sentiments

        topics = tuple(st.multiselect(
            "Topic", options=options.topics,
            default=[v for v in saved.topics if v in options.topics],
            key=f"{key_prefix}_topics",
        )) if options.topics else saved.topics

        aspects = tuple(st.multiselect(
            "Aspect", options=options.aspects,
            default=[v for v in saved.aspects if v in options.aspects],
            key=f"{key_prefix}_aspects",
        )) if options.aspects else saved.aspects

        search_text = st.text_input(
            "Search review text",
            value=saved.search_text,
            key=f"{key_prefix}_search",
        )

    filters = DashboardFilters(
        start_date=start_date,
        end_date=end_date,
        products=products,
        categories=categories,
        rating_min=rating_min,
        rating_max=rating_max,
        sentiments=sentiments,
        topics=topics,
        aspects=aspects,
        search_text=search_text,
    )
    state["active_filters"] = filters.as_dict()
    return filters


def render_filter_status(total_rows: int, filtered_rows: int, filters: DashboardFilters) -> None:
    """Tell the user whether downstream visuals use complete or filtered results."""
    if filters.is_active():
        st.caption(f"Showing {filtered_rows:,} of {total_rows:,} reviews after global filtering.")
    else:
        st.caption(f"Showing all {total_rows:,} reviews.")


def render_empty_filtered_state() -> None:
    st.warning("No reviews match the active filters. Reset the filters to continue exploring results.")


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8")
