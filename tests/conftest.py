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


@pytest.fixture
def topic_sentiment_df() -> pd.DataFrame:
    """Sentiment-enriched canonical data with three clear discussion themes."""
    rows: list[dict[str, object]] = []
    themes = [
        (
            "battery",
            [
                "battery charge lasts all day and power usage is efficient",
                "battery drains quickly and charging takes too long",
                "power and battery performance improved after the update",
                "charging speed is good but battery capacity feels average",
                "battery life is terrible when using navigation",
            ],
        ),
        (
            "shipping",
            [
                "shipping was fast and the package arrived early",
                "delivery was delayed and the package arrived damaged",
                "shipping tracking worked well and delivery was smooth",
                "package quality was fine but delivery took several days",
                "shipping service lost my package and delivery support was poor",
            ],
        ),
        (
            "support",
            [
                "customer support solved my problem immediately",
                "support agent was unhelpful and service was frustrating",
                "customer service answered quickly and fixed the account issue",
                "support response was average but the agent was polite",
                "service team never replied and customer support was disappointing",
            ],
        ),
    ]
    sentiment_cycle = ["Positive", "Negative", "Positive", "Neutral", "Negative"]
    score_cycle = [0.95, 0.92, 0.90, 0.70, 0.88]

    review_index = 0
    for theme, examples in themes:
        for repeat in range(2):
            for offset, text in enumerate(examples):
                unique_text = f"{text} example {repeat} {theme}"
                rows.append(
                    {
                        "review_id": f"review_{review_index:03d}",
                        "review_text": unique_text,
                        "clean_text": unique_text,
                        "sentiment_label": sentiment_cycle[offset],
                        "sentiment_score": score_cycle[offset],
                        "product": "Demo Product",
                    }
                )
                review_index += 1
    return pd.DataFrame(rows)
