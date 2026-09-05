from __future__ import annotations

import pandas as pd

from src.dashboard.state import (
    DEFAULT_STATE,
    initialize_session_state,
    invalidate_after_aspects,
    invalidate_after_sentiment,
    invalidate_after_topics,
    reset_for_new_dataset,
    set_canonical_dataset,
)


def test_initialize_session_state_preserves_existing_values() -> None:
    state: dict[str, object] = {"analysis_complete": True}
    initialize_session_state(state)
    assert state["analysis_complete"] is True
    assert set(DEFAULT_STATE).issubset(state)
    assert state["active_filters"] == {}


def test_reset_for_new_dataset_only_resets_when_signature_changes() -> None:
    state: dict[str, object] = {}
    initialize_session_state(state)
    state["results_df"] = pd.DataFrame({"x": [1]})
    state["source_signature"] = "same"

    changed = reset_for_new_dataset(state, "same", uploaded_file_name="same.csv")
    assert changed is False
    assert isinstance(state["results_df"], pd.DataFrame)

    changed = reset_for_new_dataset(state, "new", uploaded_file_name="new.csv")
    assert changed is True
    assert state["results_df"] is None
    assert state["source_signature"] == "new"
    assert state["uploaded_file_name"] == "new.csv"


def test_set_canonical_dataset_copies_data_and_clears_stale_outputs() -> None:
    state: dict[str, object] = {}
    initialize_session_state(state)
    state["results_df"] = pd.DataFrame({"old": [1]})
    source = pd.DataFrame({"review_text": ["hello"], "clean_text": ["hello"]})

    set_canonical_dataset(
        state,
        source,
        column_mapping={"text": "review"},
        ingestion_statistics={"input_rows": 1, "output_rows": 1},
    )
    source.loc[0, "review_text"] = "mutated"

    assert state["results_df"] is None
    assert state["sentiment_complete"] is False
    assert state["canonical_df"].loc[0, "review_text"] == "hello"  # type: ignore[index]
    assert state["clean_df"].loc[0, "review_text"] == "hello"  # type: ignore[index]


def test_stage_invalidation_only_clears_downstream_state() -> None:
    state: dict[str, object] = {}
    initialize_session_state(state)
    state.update(
        {
            "sentiment_complete": True,
            "selected_sentiment_model": "distilbert",
            "topic_complete": True,
            "aspect_complete": True,
            "insight_complete": True,
            "analysis_complete": True,
        }
    )

    invalidate_after_sentiment(state)
    assert state["sentiment_complete"] is True
    assert state["topic_complete"] is False
    assert state["aspect_complete"] is False
    assert state["insight_complete"] is False

    state["topic_complete"] = True
    state["aspect_complete"] = True
    state["insight_complete"] = True
    invalidate_after_topics(state)
    assert state["topic_complete"] is True
    assert state["aspect_complete"] is False
    assert state["insight_complete"] is False

    state["aspect_complete"] = True
    state["insight_complete"] = True
    invalidate_after_aspects(state)
    assert state["aspect_complete"] is True
    assert state["insight_complete"] is False
