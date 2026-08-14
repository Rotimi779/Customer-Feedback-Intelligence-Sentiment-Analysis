"""Tests for the unified sentiment inference contract."""

import pandas as pd

from src.pipeline import run_sentiment_stage
from src.sentiment import SentimentAnalyzer, SentimentModelName
from src.sentiment.baseline import BaselineSentimentModel


def _canonical_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "review_text": [
                "I absolutely love this product",
                "It is fine and works as expected",
                "This is a terrible product",
            ],
            "clean_text": [
                "I absolutely love this product",
                "It is fine and works as expected",
                "This is a terrible product",
            ],
        }
    )


def test_analyzer_enriches_copy_without_mutating_input(labeled_sentiment_df: pd.DataFrame) -> None:
    model = BaselineSentimentModel().fit(
        labeled_sentiment_df["review_text"],
        labeled_sentiment_df["sentiment_label"],
    )
    analyzer = SentimentAnalyzer(SentimentModelName.LOGISTIC_REGRESSION, model)
    source = _canonical_dataframe()
    result = analyzer.predict_dataframe(source)
    assert "sentiment_label" not in source.columns
    assert {"sentiment_label", "sentiment_score"}.issubset(result.dataframe.columns)
    assert len(result.dataframe) == len(source)
    assert result.dataframe["sentiment_score"].between(0, 1).all()


def test_pipeline_sentiment_stage_accepts_injected_analyzer(labeled_sentiment_df: pd.DataFrame) -> None:
    model = BaselineSentimentModel().fit(
        labeled_sentiment_df["review_text"],
        labeled_sentiment_df["sentiment_label"],
    )
    analyzer = SentimentAnalyzer(SentimentModelName.LOGISTIC_REGRESSION, model)
    result = run_sentiment_stage(
        _canonical_dataframe(),
        SentimentModelName.LOGISTIC_REGRESSION,
        analyzer=analyzer,
    )
    assert list(result.dataframe.columns)[-2:] == ["sentiment_label", "sentiment_score"]
