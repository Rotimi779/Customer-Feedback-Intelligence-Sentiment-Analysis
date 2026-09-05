"""Interactive, model-free exploratory analysis of the canonical dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.dashboard.components import (
    render_empty_filtered_state,
    render_filter_status,
    render_global_filters,
    render_prerequisite,
)
from src.dashboard.errors import check_page_prerequisites
from src.dashboard.filters import apply_dashboard_filters
from src.dashboard.state import initialize_session_state

from src.eda import (
    DatasetFilters,
    apply_filters,
    build_common_words_chart,
    build_dataset_summary,
    build_group_distribution_chart,
    build_missing_values_chart,
    build_rating_distribution_chart,
    build_review_length_histogram,
    build_reviews_over_time_chart,
    format_bytes,
    get_filter_options,
    missing_value_summary,
    summarize_quality,
)


APP_TITLE = "Dataset Overview"


def _clear_filter_state(keys: tuple[str, ...]) -> None:
    """Remove widget state before Streamlit recreates the filter controls."""
    for key in keys:
        st.session_state.pop(key, None)


@st.cache_data(show_spinner=False)
def _cached_apply_filters(
    dataframe: pd.DataFrame,
    filters: DatasetFilters,
) -> pd.DataFrame:
    """Cache deterministic filtering across normal UI reruns."""
    return apply_filters(dataframe, filters)


@st.cache_data(show_spinner=False)
def _cached_eda_bundle(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Calculate all metrics and charts for one filtered DataFrame."""
    return {
        "summary": build_dataset_summary(dataframe),
        "quality": summarize_quality(dataframe),
        "missing_table": missing_value_summary(dataframe),
        "review_length_figure": build_review_length_histogram(dataframe),
        "common_words_figure": build_common_words_chart(dataframe),
        "missing_values_figure": build_missing_values_chart(dataframe),
        "timeline_figure": build_reviews_over_time_chart(dataframe),
        "rating_figure": build_rating_distribution_chart(dataframe),
        "product_figure": build_group_distribution_chart(
            dataframe,
            column="product",
        ),
        "category_figure": build_group_distribution_chart(
            dataframe,
            column="category",
        ),
    }


def _render_sidebar_filters(dataframe: pd.DataFrame) -> DatasetFilters:
    """Build only the filters supported by available optional metadata."""
    options = get_filter_options(dataframe)
    signature = str(st.session_state.get("source_signature") or "dataset")[:12]
    prefix = f"eda_{signature}"
    widget_keys = (
        f"{prefix}_start_date",
        f"{prefix}_end_date",
        f"{prefix}_products",
        f"{prefix}_categories",
        f"{prefix}_ratings",
        f"{prefix}_keyword",
    )

    with st.sidebar:
        st.header("Dataset filters")
        st.button(
            "Clear filters",
            use_container_width=True,
            on_click=_clear_filter_state,
            args=(widget_keys,),
        )

        start_date = None
        end_date = None
        if options.minimum_date is not None and options.maximum_date is not None:
            start_date = st.date_input(
                "Start date",
                value=options.minimum_date.date(),
                min_value=options.minimum_date.date(),
                max_value=options.maximum_date.date(),
                key=f"{prefix}_start_date",
            )
            end_date = st.date_input(
                "End date",
                value=options.maximum_date.date(),
                min_value=options.minimum_date.date(),
                max_value=options.maximum_date.date(),
                key=f"{prefix}_end_date",
            )

        products: tuple[str, ...] = ()
        if options.products:
            products = tuple(
                st.multiselect(
                    "Products",
                    options=options.products,
                    key=f"{prefix}_products",
                )
            )

        categories: tuple[str, ...] = ()
        if options.categories:
            categories = tuple(
                st.multiselect(
                    "Categories",
                    options=options.categories,
                    key=f"{prefix}_categories",
                )
            )

        rating_min = None
        rating_max = None
        if (
            options.minimum_rating is not None
            and options.maximum_rating is not None
        ):
            if options.minimum_rating < options.maximum_rating:
                rating_min, rating_max = st.slider(
                    "Rating range",
                    min_value=float(options.minimum_rating),
                    max_value=float(options.maximum_rating),
                    value=(
                        float(options.minimum_rating),
                        float(options.maximum_rating),
                    ),
                    key=f"{prefix}_ratings",
                )
            else:
                st.caption(f"All reviews have rating {options.minimum_rating:g}.")

        keyword = st.text_input(
            "Keyword search",
            placeholder="Search review text",
            key=f"{prefix}_keyword",
        )
        st.caption("Every chart, metric, and preview uses these filters.")

    return DatasetFilters(
        start_date=start_date,
        end_date=end_date,
        products=products,
        categories=categories,
        rating_min=rating_min,
        rating_max=rating_max,
        keyword=keyword,
    )


def _render_kpis(summary: Any, *, total_unfiltered: int) -> None:
    """Render required dataset-composition metrics."""
    retention = 0.0 if total_unfiltered == 0 else summary.total_reviews / total_unfiltered
    row_1 = st.columns(6)
    row_1[0].metric(
        "Reviews",
        f"{summary.total_reviews:,}",
        help=f"{retention:.1%} of the prepared dataset after filters.",
    )
    row_1[1].metric("Duplicate reviews", f"{summary.duplicate_reviews:,}")
    row_1[2].metric("Missing reviews", f"{summary.missing_reviews:,}")
    row_1[3].metric(
        "Average length",
        f"{summary.average_review_length:.1f} words",
    )
    row_1[4].metric(
        "Median length",
        f"{summary.median_review_length:.1f} words",
    )
    row_1[5].metric("Dataset size", format_bytes(summary.dataset_size_bytes))

    optional_metrics: list[tuple[str, str]] = []
    if summary.average_rating is not None:
        optional_metrics.append(("Average rating", f"{summary.average_rating:.2f}"))
    if summary.minimum_date is not None and summary.maximum_date is not None:
        optional_metrics.append(
            (
                "Date range",
                (
                    f"{summary.minimum_date.date().isoformat()} to "
                    f"{summary.maximum_date.date().isoformat()}"
                ),
            )
        )
    if summary.product_count is not None:
        optional_metrics.append(("Products", f"{summary.product_count:,}"))
    if summary.category_count is not None:
        optional_metrics.append(("Categories", f"{summary.category_count:,}"))

    if optional_metrics:
        columns = st.columns(len(optional_metrics))
        for column, (label, value) in zip(columns, optional_metrics, strict=True):
            column.metric(label, value)


def _render_source_quality(
    bundle: dict[str, Any],
    ingestion_statistics: dict[str, int] | None,
) -> None:
    """Show source-cleaning audit metrics and current missing-value detail."""
    quality = bundle["quality"]
    ingestion_statistics = ingestion_statistics or {}

    st.subheader("Dataset quality")
    quality_columns = st.columns(5)
    quality_columns[0].metric(
        "Input rows",
        f"{ingestion_statistics.get('input_rows', quality.total_rows):,}",
    )
    quality_columns[1].metric(
        "Empty rows removed",
        f"{ingestion_statistics.get('empty_reviews_removed', 0):,}",
    )
    quality_columns[2].metric(
        "Duplicates removed",
        f"{ingestion_statistics.get('duplicate_reviews_removed', 0):,}",
    )
    quality_columns[3].metric("Missing cells", f"{quality.missing_cells:,}")
    quality_columns[4].metric(
        "Completeness",
        f"{quality.completeness_percentage:.1f}%",
    )

    chart_column, table_column = st.columns((1.4, 1.0))
    with chart_column:
        figure = bundle["missing_values_figure"]
        if figure is None:
            st.info("No columns are available for missing-value analysis.")
        else:
            st.plotly_chart(figure, use_container_width=True)
    with table_column:
        missing_table = bundle["missing_table"].copy()
        if missing_table.empty:
            st.info("No missing-value details are available.")
        else:
            missing_table["missing_percentage"] = missing_table[
                "missing_percentage"
            ].round(1)
            st.dataframe(
                missing_table,
                use_container_width=True,
                hide_index=True,
            )


def _render_charts(bundle: dict[str, Any]) -> None:
    """Render required and metadata-dependent visualizations."""
    st.subheader("Text characteristics")
    left, right = st.columns(2)
    with left:
        figure = bundle["review_length_figure"]
        if figure is None:
            st.info("Review-length data is unavailable for the current filters.")
        else:
            st.plotly_chart(figure, use_container_width=True)
    with right:
        figure = bundle["common_words_figure"]
        if figure is None:
            st.info("No common words are available for the current filters.")
        else:
            st.plotly_chart(figure, use_container_width=True)

    optional_figures = [
        bundle["timeline_figure"],
        bundle["rating_figure"],
        bundle["product_figure"],
        bundle["category_figure"],
    ]
    available_figures = [figure for figure in optional_figures if figure is not None]
    if not available_figures:
        return

    st.subheader("Metadata patterns")
    for index in range(0, len(available_figures), 2):
        columns = st.columns(2)
        for offset, figure in enumerate(available_figures[index : index + 2]):
            with columns[offset]:
                st.plotly_chart(figure, use_container_width=True)


def main() -> None:
    """Render the complete Phase 3 EDA Overview page."""
    st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
    initialize_session_state(st.session_state)
    st.title(APP_TITLE)
    st.caption(
        "Understand dataset quality and structure before any NLP model inference."
    )

    status = check_page_prerequisites(st.session_state, "overview")
    if not render_prerequisite(status):
        return
    clean_df = st.session_state.get("canonical_df")
    if not isinstance(clean_df, pd.DataFrame):
        clean_df = st.session_state.get("clean_df")
    if clean_df.empty:
        st.warning("The prepared dataset has no usable reviews.")
        return

    filters = render_global_filters(
        clean_df, st.session_state, key_prefix="overview_global"
    )
    filtered_df = apply_dashboard_filters(clean_df, filters)
    render_filter_status(len(clean_df), len(filtered_df), filters)

    if filtered_df.empty:
        render_empty_filtered_state()
        return

    bundle = _cached_eda_bundle(filtered_df)
    _render_kpis(bundle["summary"], total_unfiltered=len(clean_df))

    st.divider()
    _render_source_quality(
        bundle,
        st.session_state.get("ingestion_statistics"),
    )

    st.divider()
    _render_charts(bundle)

    st.divider()
    st.subheader("Filtered data preview")
    st.dataframe(
        filtered_df.head(200),
        use_container_width=True,
        hide_index=True,
    )
    if len(filtered_df) > 200:
        st.caption("Preview limited to the first 200 matching reviews.")


if __name__ == "__main__":
    main()
