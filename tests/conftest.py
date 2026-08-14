"""Pytest configuration for repository-local imports."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest


@pytest.fixture
def labeled_sentiment_df() -> pd.DataFrame:
    """Balanced three-class data large enough for 70/15/15 stratified tests."""
    rows: list[dict[str, str]] = []
    for index in range(15):
        rows.append(
            {
                "review_text": f"Excellent product experience number {index}; I love how reliable it is.",
                "sentiment_label": "Positive",
            }
        )
        rows.append(
            {
                "review_text": f"Average product experience number {index}; it works as expected.",
                "sentiment_label": "Neutral",
            }
        )
        rows.append(
            {
                "review_text": f"Terrible product experience number {index}; I hate how unreliable it is.",
                "sentiment_label": "Negative",
            }
        )
    return pd.DataFrame(rows)
