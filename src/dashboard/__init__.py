"""Shared dashboard integration utilities."""

from src.dashboard.errors import PrerequisiteStatus, check_page_prerequisites
from src.dashboard.filters import (
    DashboardFilterOptions,
    DashboardFilters,
    apply_dashboard_filters,
    get_dashboard_filter_options,
)
from src.dashboard.formatting import (
    SENTIMENT_COLORS,
    SENTIMENT_ORDER,
    format_confidence,
    format_date,
    format_label,
    format_large_number,
    format_percentage,
    truncate_text,
)
from src.dashboard.state import (
    DEFAULT_STATE,
    current_results_dataframe,
    initialize_session_state,
    invalidate_after_aspects,
    invalidate_after_sentiment,
    invalidate_after_topics,
    reset_analysis_state,
    reset_for_new_dataset,
    set_canonical_dataset,
)
from src.dashboard.workflow import FullAnalysisResult, persist_full_analysis, run_full_analysis

__all__ = [
    "DEFAULT_STATE",
    "DashboardFilterOptions",
    "DashboardFilters",
    "FullAnalysisResult",
    "PrerequisiteStatus",
    "SENTIMENT_COLORS",
    "SENTIMENT_ORDER",
    "apply_dashboard_filters",
    "check_page_prerequisites",
    "current_results_dataframe",
    "format_confidence",
    "format_date",
    "format_label",
    "format_large_number",
    "format_percentage",
    "get_dashboard_filter_options",
    "initialize_session_state",
    "invalidate_after_aspects",
    "invalidate_after_sentiment",
    "invalidate_after_topics",
    "persist_full_analysis",
    "reset_analysis_state",
    "reset_for_new_dataset",
    "run_full_analysis",
    "set_canonical_dataset",
    "truncate_text",
]
