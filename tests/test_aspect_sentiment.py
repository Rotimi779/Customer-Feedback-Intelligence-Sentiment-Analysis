"""Tests for review-sentiment reuse across detected aspects."""

import pandas as pd

from src.aspects.sentiment import associate_aspect_sentiment


def test_aspect_sentiment_enriches_copy_without_mutating_input(
    aspect_topic_df: pd.DataFrame,
) -> None:
    original = aspect_topic_df.copy(deep=True)
    enriched = associate_aspect_sentiment(aspect_topic_df)

    pd.testing.assert_frame_equal(aspect_topic_df, original)
    assert {"detected_aspects", "aspect_sentiment", "aspect_confidence"}.issubset(
        enriched.columns
    )
    assert enriched.loc[0, "aspect_sentiment"]["Battery"] == "Positive"
    assert enriched.loc[0, "aspect_confidence"]["Battery"] == 0.96


def test_multiple_aspects_receive_same_review_level_sentiment(
    aspect_topic_df: pd.DataFrame,
) -> None:
    enriched = associate_aspect_sentiment(aspect_topic_df)
    mapping = enriched.loc[2, "aspect_sentiment"]

    assert mapping["Customer Support"] == "Neutral"
    assert mapping["User Interface"] == "Neutral"
    # This is the documented MVP limitation: review sentiment is reused.
    assert len(set(mapping.values())) == 1
