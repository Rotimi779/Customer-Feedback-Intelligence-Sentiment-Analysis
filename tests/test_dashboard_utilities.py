from __future__ import annotations

import pandas as pd

from src.dashboard.errors import check_page_prerequisites
from src.dashboard.formatting import (
    SENTIMENT_COLORS,
    format_confidence,
    format_label,
    format_large_number,
    format_percentage,
    truncate_text,
)


def test_sentiment_colors_are_semantically_stable() -> None:
    assert SENTIMENT_COLORS["Positive"] == "#2E8B57"
    assert SENTIMENT_COLORS["Neutral"] == "#808080"
    assert SENTIMENT_COLORS["Negative"] == "#D9534F"


def test_formatting_helpers_are_consistent() -> None:
    assert format_percentage(0.456) == "45.6%"
    assert format_confidence(0.932) == "93.2%"
    assert format_large_number(12_400) == "12.4K"
    assert format_label("sentiment_label") == "Sentiment Label"
    assert truncate_text("abcdef", max_length=5) == "abcd…"


def test_page_prerequisites_follow_pipeline_order() -> None:
    canonical = pd.DataFrame({"review_text": ["x"], "clean_text": ["x"]})
    state: dict[str, object] = {"canonical_df": canonical}

    sentiment_status = check_page_prerequisites(state, "sentiment")
    assert not sentiment_status.ready
    assert sentiment_status.next_page == "app.py"

    state["sentiment_complete"] = True
    assert check_page_prerequisites(state, "sentiment").ready

    topic_status = check_page_prerequisites(state, "topics")
    assert not topic_status.ready
    assert topic_status.next_page == "app.py"

    state["topic_complete"] = True
    assert check_page_prerequisites(state, "topics").ready

    assert not check_page_prerequisites(state, "aspects").ready

    state["aspect_complete"] = True
    assert check_page_prerequisites(state, "aspects").ready

    assert not check_page_prerequisites(state, "insights").ready

    state["insight_complete"] = True
    assert check_page_prerequisites(state, "insights").ready
