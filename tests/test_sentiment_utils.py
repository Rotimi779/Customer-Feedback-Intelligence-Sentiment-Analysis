"""Tests for labelled-data preparation and reproducible splits."""

import pandas as pd
import pytest

from src.sentiment.utils import (
    SENTIMENT_LABELS,
    SentimentDataError,
    normalize_sentiment_label,
    prepare_labeled_dataframe,
    split_labeled_dataframe,
)


def test_normalize_sentiment_label_accepts_canonical_case_variants() -> None:
    assert normalize_sentiment_label("positive") == "Positive"
    assert normalize_sentiment_label("NEUTRAL") == "Neutral"


def test_normalize_sentiment_label_uses_only_explicit_custom_mapping() -> None:
    assert normalize_sentiment_label("2", {"2": "Positive"}) == "Positive"
    with pytest.raises(SentimentDataError):
        normalize_sentiment_label("2")


def test_prepare_labeled_dataframe_removes_empty_and_duplicate_text(
    labeled_sentiment_df: pd.DataFrame,
) -> None:
    extra = pd.DataFrame(
        [
            {"review_text": "", "sentiment_label": "Positive"},
            labeled_sentiment_df.iloc[0].to_dict(),
        ]
    )
    prepared = prepare_labeled_dataframe(pd.concat([labeled_sentiment_df, extra], ignore_index=True))
    assert len(prepared) == len(labeled_sentiment_df)
    assert set(prepared["sentiment_label"]) == set(SENTIMENT_LABELS)


def test_split_labeled_dataframe_is_reproducible(labeled_sentiment_df: pd.DataFrame) -> None:
    prepared = prepare_labeled_dataframe(labeled_sentiment_df)
    first = split_labeled_dataframe(prepared)
    second = split_labeled_dataframe(prepared)
    assert [part["review_text"].tolist() for part in first] == [
        part["review_text"].tolist() for part in second
    ]
    assert sum(len(part) for part in first) == len(prepared)
    for part in first:
        assert set(part["sentiment_label"]) == set(SENTIMENT_LABELS)
