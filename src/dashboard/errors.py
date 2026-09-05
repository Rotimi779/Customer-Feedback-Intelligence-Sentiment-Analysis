"""Prerequisite and user-facing dashboard state checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class PrerequisiteStatus:
    ready: bool
    message: str | None = None
    next_page: str | None = None


def _has_dataframe(state: Mapping[str, Any], key: str) -> bool:
    value = state.get(key)
    return isinstance(value, pd.DataFrame) and not value.empty


def check_page_prerequisites(state: Mapping[str, Any], page: str) -> PrerequisiteStatus:
    """Return whether the saved dashboard state can render one analysis page."""
    normalized = page.strip().casefold().replace(" ", "_")

    has_canonical = _has_dataframe(state, "canonical_df") or _has_dataframe(state, "clean_df")
    if normalized in {"overview", "sentiment", "sentiment_analysis"}:
        if not has_canonical:
            return PrerequisiteStatus(False, "Upload and confirm a dataset first.", "app.py")
        return PrerequisiteStatus(True)

    if normalized in {"topics", "topic_modeling"}:
        if not has_canonical:
            return PrerequisiteStatus(False, "Upload and confirm a dataset first.", "app.py")
        if not state.get("sentiment_complete"):
            return PrerequisiteStatus(False, "Run sentiment analysis before exploring topics.", "pages/2_Sentiment.py")
        return PrerequisiteStatus(True)

    if normalized in {"aspects", "aspect_analysis"}:
        if not state.get("topic_complete"):
            return PrerequisiteStatus(False, "Run topic modeling before aspect analysis.", "pages/3_Topics.py")
        return PrerequisiteStatus(True)

    if normalized in {"insights", "business_insights"}:
        if not state.get("aspect_complete"):
            return PrerequisiteStatus(False, "Run aspect analysis before generating business insights.", "pages/4_Aspect_Analysis.py")
        return PrerequisiteStatus(True)

    if normalized in {"data_explorer", "explorer"}:
        if not has_canonical:
            return PrerequisiteStatus(False, "Upload and confirm a dataset first.", "app.py")
        return PrerequisiteStatus(True)

    raise ValueError(f"Unknown dashboard page '{page}'.")
