"""Shared Streamlit session-state contract for the dashboard integration phase."""

from __future__ import annotations

from collections.abc import MutableMapping
from copy import deepcopy
from typing import Any

import pandas as pd


DEFAULT_STATE: dict[str, Any] = {
    "uploaded_file_name": None,
    "raw_df": None,
    "clean_df": None,
    "canonical_df": None,
    "filtered_df": None,
    "column_mapping": None,
    "validation_report": None,
    "ingestion_statistics": None,
    "source_signature": None,
    "analysis_complete": False,
    "analysis_running": False,
    "analysis_stage": None,
    "last_error": None,
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
    "active_filters": {},
    "selected_topic": None,
    "selected_aspect": None,
}

DERIVED_KEYS: tuple[str, ...] = (
    "filtered_df",
    "analysis_complete",
    "analysis_running",
    "analysis_stage",
    "last_error",
    "results_df",
    "sentiment_complete",
    "selected_sentiment_model",
    "sentiment_runtime_seconds",
    "sentiment_source_signature",
    "topic_summary",
    "topic_complete",
    "topic_metrics",
    "topic_source_signature",
    "topic_config",
    "topic_model_runtime",
    "topic_representatives",
    "aspect_summary",
    "aspect_mentions",
    "aspect_metrics",
    "aspect_complete",
    "aspect_source_signature",
    "aspect_runtime_seconds",
    "insights",
    "insight_complete",
    "insight_source_signature",
    "insight_runtime_seconds",
    "active_filters",
    "selected_topic",
    "selected_aspect",
)


def _default_value(key: str) -> Any:
    value = DEFAULT_STATE[key]
    if isinstance(value, (dict, list, set)):
        return deepcopy(value)
    return value


def initialize_session_state(state: MutableMapping[str, Any]) -> None:
    """Populate missing dashboard keys without replacing existing values."""
    for key in DEFAULT_STATE:
        if key not in state:
            state[key] = _default_value(key)


def reset_analysis_state(state: MutableMapping[str, Any]) -> None:
    """Clear all analysis outputs while preserving the active uploaded dataset."""
    initialize_session_state(state)
    for key in DERIVED_KEYS:
        state[key] = _default_value(key)


def reset_for_new_dataset(
    state: MutableMapping[str, Any],
    source_signature: str,
    *,
    uploaded_file_name: str | None = None,
) -> bool:
    """Reset dataset + downstream state only when the active file changes.

    Returns ``True`` when a reset occurred.
    """
    initialize_session_state(state)
    if state.get("source_signature") == source_signature:
        if uploaded_file_name is not None:
            state["uploaded_file_name"] = uploaded_file_name
        return False

    for key in DEFAULT_STATE:
        state[key] = _default_value(key)
    state["source_signature"] = source_signature
    state["uploaded_file_name"] = uploaded_file_name
    return True


def set_canonical_dataset(
    state: MutableMapping[str, Any],
    dataframe: pd.DataFrame,
    *,
    column_mapping: dict[str, Any],
    ingestion_statistics: dict[str, Any],
    validation_report: Any = None,
) -> None:
    """Store a confirmed canonical DataFrame and invalidate stale analysis."""
    initialize_session_state(state)
    reset_analysis_state(state)
    canonical = dataframe.copy()
    state["clean_df"] = canonical
    state["canonical_df"] = canonical
    state["column_mapping"] = deepcopy(column_mapping)
    state["ingestion_statistics"] = deepcopy(ingestion_statistics)
    state["validation_report"] = validation_report


def invalidate_after_sentiment(state: MutableMapping[str, Any]) -> None:
    """Invalidate topic/aspect/insight outputs after sentiment is rerun."""
    initialize_session_state(state)
    for key in (
        "topic_summary", "topic_complete", "topic_metrics", "topic_source_signature",
        "topic_config", "topic_model_runtime", "topic_representatives",
        "aspect_summary", "aspect_mentions", "aspect_metrics", "aspect_complete",
        "aspect_source_signature", "aspect_runtime_seconds", "insights",
        "insight_complete", "insight_source_signature", "insight_runtime_seconds",
        "analysis_complete", "filtered_df", "active_filters", "selected_topic",
        "selected_aspect",
    ):
        state[key] = _default_value(key)


def invalidate_after_topics(state: MutableMapping[str, Any]) -> None:
    """Invalidate aspect/insight outputs after topic modeling is rerun."""
    initialize_session_state(state)
    for key in (
        "aspect_summary", "aspect_mentions", "aspect_metrics", "aspect_complete",
        "aspect_source_signature", "aspect_runtime_seconds", "insights",
        "insight_complete", "insight_source_signature", "insight_runtime_seconds",
        "analysis_complete", "filtered_df", "active_filters", "selected_aspect",
    ):
        state[key] = _default_value(key)


def invalidate_after_aspects(state: MutableMapping[str, Any]) -> None:
    """Invalidate insights after aspect analysis is rerun."""
    initialize_session_state(state)
    for key in (
        "insights", "insight_complete", "insight_source_signature",
        "insight_runtime_seconds", "analysis_complete", "filtered_df",
        "active_filters",
    ):
        state[key] = _default_value(key)


def current_results_dataframe(state: MutableMapping[str, Any]) -> pd.DataFrame | None:
    """Return the most enriched saved DataFrame, falling back to canonical data."""
    initialize_session_state(state)
    results = state.get("results_df")
    if isinstance(results, pd.DataFrame):
        return results
    canonical = state.get("canonical_df")
    if isinstance(canonical, pd.DataFrame):
        return canonical
    clean = state.get("clean_df")
    return clean if isinstance(clean, pd.DataFrame) else None
