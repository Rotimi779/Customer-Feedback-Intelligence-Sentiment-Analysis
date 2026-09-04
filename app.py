"""Streamlit upload and data-ingestion workflow for the MVP."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from src.ingestion import (
    CSVLoadError,
    DEFAULT_INGESTION_CONFIG,
    ColumnMapping,
    ValidationReport,
    ValidationSeverity,
    detect_metadata_columns,
    detect_text_column,
    load_csv,
    normalize_column_names,
    rank_text_columns,
    validate_dataset,
)
from src.pipeline import prepare_dataset

APP_TITLE = "AI Customer Feedback Intelligence Platform"
PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample" / "sample_customer_reviews.csv"
NOT_MAPPED = "Not mapped"


def configure_logging() -> logging.Logger:
    """Configure and return the application logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(__name__)


def initialize_session_state() -> None:
    """Create the shared state contract defined by the technical design."""
    defaults = {
        "raw_df": None,
        "clean_df": None,
        "column_mapping": None,
        "analysis_complete": False,
        "results_df": None,
        "sentiment_complete": False,
        "selected_sentiment_model": None,
        "sentiment_runtime_seconds": None,
        "sentiment_source_signature": None,
        "topic_summary": None,
        "topic_complete": False,
        "topic_metrics": None,
        "topic_source_signature": None,
        "topic_config": None,
        "topic_model_runtime": None,
        "topic_representatives": None,
        "aspect_summary": None,
        "aspect_mentions": None,
        "aspect_metrics": None,
        "aspect_complete": False,
        "aspect_source_signature": None,
        "aspect_runtime_seconds": None,
        "insights": None,
        "insight_complete": False,
        "insight_source_signature": None,
        "insight_runtime_seconds": None,
        "source_signature": None,
        "ingestion_statistics": None,
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def reset_dataset_state(source_signature: str) -> None:
    """Clear derived state when a different CSV becomes active."""
    if st.session_state["source_signature"] == source_signature:
        return

    st.session_state["source_signature"] = source_signature
    st.session_state["raw_df"] = None
    st.session_state["clean_df"] = None
    st.session_state["column_mapping"] = None
    st.session_state["analysis_complete"] = False
    st.session_state["results_df"] = None
    st.session_state["sentiment_complete"] = False
    st.session_state["selected_sentiment_model"] = None
    st.session_state["sentiment_runtime_seconds"] = None
    st.session_state["sentiment_source_signature"] = None
    st.session_state["topic_summary"] = None
    st.session_state["topic_complete"] = False
    st.session_state["topic_metrics"] = None
    st.session_state["topic_source_signature"] = None
    st.session_state["topic_config"] = None
    st.session_state["topic_model_runtime"] = None
    st.session_state["topic_representatives"] = None
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
    st.session_state["ingestion_statistics"] = None


@st.cache_data(show_spinner=False)
def load_csv_bytes(payload: bytes, filename: str):
    """Load uploaded bytes once across normal Streamlit reruns."""
    return load_csv(
        payload,
        filename=filename,
        config=DEFAULT_INGESTION_CONFIG,
    )


def render_sidebar() -> None:
    """Render navigation and the active implementation phase."""
    with st.sidebar:
        st.header("Navigation")
        st.page_link("app.py", label="Upload & Setup", icon="📤")
        st.page_link("pages/1_Overview.py", label="Overview", icon="📊")
        st.page_link("pages/2_Sentiment.py", label="Sentiment", icon="🙂")
        st.page_link("pages/3_Topics.py", label="Topics", icon="🧩")
        st.page_link(
            "pages/4_Aspect_Analysis.py",
            label="Aspect Analysis",
            icon="🔎",
        )
        st.page_link("pages/5_Insights.py", label="Insights", icon="💡")
        st.page_link(
            "pages/6_Data_Explorer.py",
            label="Data Explorer",
            icon="🗂️",
        )
        st.divider()
        st.caption("Phase 7: business insights")


def render_validation_report(report: ValidationReport) -> None:
    """Display validation messages without exposing internal exceptions."""
    if not report.messages:
        st.success("Dataset validation passed with no warnings.")
        return

    for item in report.messages:
        message = item.message
        if item.remediation:
            message = f"{message} {item.remediation}"

        if item.severity is ValidationSeverity.ERROR:
            st.error(message)
        elif item.severity is ValidationSeverity.WARNING:
            st.warning(message)
        else:
            st.info(message)


def optional_mapping_select(
    canonical_name: str,
    columns: list[str],
    suggested: str | None,
    source_signature: str,
) -> str | None:
    """Render one optional metadata mapping selector."""
    options = [NOT_MAPPED, *columns]
    default_value = suggested if suggested in columns else NOT_MAPPED
    selected = st.selectbox(
        canonical_name.replace("_", " ").title(),
        options=options,
        index=options.index(default_value),
        key=f"metadata_{canonical_name}_{source_signature[:12]}",
        help=f"Optional source column to store internally as '{canonical_name}'.",
    )
    return None if selected == NOT_MAPPED else selected


def render_canonical_result(mapping: ColumnMapping) -> None:
    """Show a completed ingestion result only when it matches current mapping."""
    clean_df = st.session_state["clean_df"]
    stored_mapping = st.session_state["column_mapping"]
    statistics = st.session_state["ingestion_statistics"]

    if clean_df is None or stored_mapping != mapping.as_dict() or statistics is None:
        return

    st.divider()
    st.subheader("Canonical dataset")
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Input rows", f"{statistics['input_rows']:,}")
    metric_2.metric("Ready for analysis", f"{statistics['output_rows']:,}")
    metric_3.metric("Empty removed", f"{statistics['empty_reviews_removed']:,}")
    metric_4.metric("Duplicates removed", f"{statistics['duplicate_reviews_removed']:,}")

    st.dataframe(
        clean_df.head(DEFAULT_INGESTION_CONFIG.preview_rows),
        use_container_width=True,
        hide_index=True,
    )
    st.success(
        "Ingestion is complete. The canonical DataFrame is stored in session "
        "state and is ready for the Overview page."
    )


def main() -> None:
    """Render CSV upload, mapping, validation, and canonicalization."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    logger = configure_logging()
    initialize_session_state()
    render_sidebar()

    st.title(APP_TITLE)
    st.subheader("Upload customer feedback and prepare a reliable analysis dataset")
    st.caption(
        "CSV files are processed in memory. Do not upload sensitive personal data."
    )

    st.markdown("### 1. Choose a dataset")
    source_mode = st.radio(
        "Data source",
        options=("Upload a CSV", "Use bundled sample data"),
        horizontal=True,
    )

    payload: bytes | None = None
    filename: str | None = None

    if source_mode == "Upload a CSV":
        uploaded_file = st.file_uploader(
            "Customer-feedback CSV",
            type=("csv",),
            help=(
                "Maximum file size: "
                f"{DEFAULT_INGESTION_CONFIG.max_file_size_bytes / (1024 * 1024):g} MB. "
                f"Maximum rows: {DEFAULT_INGESTION_CONFIG.max_rows:,}."
            ),
        )
        if uploaded_file is not None:
            payload = uploaded_file.getvalue()
            filename = uploaded_file.name
    else:
        if not SAMPLE_DATA_PATH.exists():
            st.error("The bundled sample dataset is missing from data/sample.")
            return
        payload = SAMPLE_DATA_PATH.read_bytes()
        filename = SAMPLE_DATA_PATH.name

    if payload is None or filename is None:
        st.info("Upload a CSV to begin. No analysis runs until validation succeeds.")
        return

    source_signature = hashlib.sha256(filename.encode("utf-8") + payload).hexdigest()
    reset_dataset_state(source_signature)

    try:
        loaded = load_csv_bytes(payload, filename)
    except CSVLoadError as exc:
        logger.warning("CSV loading failed for %s: %s", filename, exc)
        st.error(str(exc))
        return
    except Exception:
        logger.exception("Unexpected CSV loading failure for %s", filename)
        st.error(
            "The CSV could not be loaded because of an unexpected error. "
            "Confirm that it is a valid UTF-8 CSV and try again."
        )
        return

    normalized_df, header_mapping = normalize_column_names(loaded.dataframe)
    st.session_state["raw_df"] = normalized_df

    logger.info(
        "Loaded CSV filename=%s rows=%s columns=%s encoding=%s size_bytes=%s",
        loaded.filename,
        len(normalized_df),
        len(normalized_df.columns),
        loaded.encoding,
        loaded.size_bytes,
    )

    st.markdown("### 2. Review the uploaded dataset")
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Rows", f"{len(normalized_df):,}")
    metric_2.metric("Columns", len(normalized_df.columns))
    metric_3.metric("Encoding", loaded.encoding)
    metric_4.metric("File size", f"{loaded.size_bytes / 1024:.1f} KB")

    with st.expander("Normalized column names", expanded=False):
        mapping_rows = [
            {"Source column": source, "Internal source name": normalized}
            for source, normalized in header_mapping.items()
        ]
        st.dataframe(pd.DataFrame(mapping_rows), use_container_width=True, hide_index=True)

    st.dataframe(
        normalized_df.head(DEFAULT_INGESTION_CONFIG.preview_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 3. Confirm column mappings")
    suggested_text = detect_text_column(normalized_df)
    candidates = rank_text_columns(normalized_df)
    columns = [str(column) for column in normalized_df.columns]
    suggested_index = columns.index(suggested_text) if suggested_text in columns else 0

    selected_text = st.selectbox(
        "Review text column",
        options=columns,
        index=suggested_index,
        help=(
            "The system suggests a column using its name, average text length, "
            "populated-value ratio, and textual content. You can override it."
        ),
    )

    if suggested_text is None:
        st.warning(
            "No text column passed the automatic confidence threshold. "
            "Select the correct review column manually."
        )
    else:
        st.caption(f"Automatic suggestion: `{suggested_text}`")

    if candidates:
        with st.expander("Text-column detection details", expanded=False):
            candidate_df = pd.DataFrame(
                [
                    {
                        "Column": candidate.column,
                        "Score": candidate.score,
                        "Average length": round(candidate.average_length, 1),
                        "Populated": f"{candidate.non_empty_ratio:.1%}",
                    }
                    for candidate in candidates
                ]
            )
            st.dataframe(candidate_df, use_container_width=True, hide_index=True)

    detected_metadata = detect_metadata_columns(
        normalized_df,
        text_column=selected_text,
    )

    with st.expander("Optional metadata mapping", expanded=True):
        st.caption(
            "Optional fields improve later charts but never block ingestion when absent."
        )
        map_col_1, map_col_2, map_col_3 = st.columns(3)
        with map_col_1:
            date_column = optional_mapping_select(
                "date",
                columns,
                detected_metadata.get("date"),
                source_signature,
            )
            rating_column = optional_mapping_select(
                "rating",
                columns,
                detected_metadata.get("rating"),
                source_signature,
            )
        with map_col_2:
            product_column = optional_mapping_select(
                "product",
                columns,
                detected_metadata.get("product"),
                source_signature,
            )
            category_column = optional_mapping_select(
                "category",
                columns,
                detected_metadata.get("category"),
                source_signature,
            )
        with map_col_3:
            version_column = optional_mapping_select(
                "version",
                columns,
                detected_metadata.get("version"),
                source_signature,
            )

    mapping = ColumnMapping(
        text=selected_text,
        date=date_column,
        rating=rating_column,
        product=product_column,
        category=category_column,
        version=version_column,
    )

    st.markdown("### 4. Validate and prepare")
    report = validate_dataset(normalized_df, mapping)
    render_validation_report(report)

    st.caption(
        "During this phase, Run Analysis validates and prepares the canonical "
        "dataset. NLP inference is added in later phases."
    )
    run_clicked = st.button(
        "Run Analysis",
        type="primary",
        disabled=not report.is_valid,
        use_container_width=True,
    )

    if run_clicked:
        try:
            result = prepare_dataset(normalized_df, mapping)
        except Exception:
            logger.exception("Canonical dataset preparation failed")
            st.error(
                "The dataset passed initial validation but could not be prepared. "
                "Review the mappings and try again."
            )
        else:
            st.session_state["clean_df"] = result.dataframe
            st.session_state["column_mapping"] = mapping.as_dict()
            st.session_state["ingestion_statistics"] = {
                "input_rows": result.statistics.input_rows,
                "output_rows": result.statistics.output_rows,
                "empty_reviews_removed": result.statistics.empty_reviews_removed,
                "duplicate_reviews_removed": result.statistics.duplicate_reviews_removed,
            }
            st.session_state["analysis_complete"] = False
            st.session_state["results_df"] = None
            st.session_state["sentiment_complete"] = False
            st.session_state["selected_sentiment_model"] = None
            st.session_state["sentiment_runtime_seconds"] = None
            st.session_state["sentiment_source_signature"] = None
            logger.info(
                "Canonicalization completed input_rows=%s output_rows=%s "
                "empty_removed=%s duplicates_removed=%s",
                result.statistics.input_rows,
                result.statistics.output_rows,
                result.statistics.empty_reviews_removed,
                result.statistics.duplicate_reviews_removed,
            )
            for warning in result.warnings:
                st.warning(warning.message)

    render_canonical_result(mapping)


if __name__ == "__main__":
    main()
