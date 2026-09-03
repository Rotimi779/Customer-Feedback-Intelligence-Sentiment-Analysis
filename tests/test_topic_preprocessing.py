"""Tests for topic-specific text preparation."""

import pandas as pd

from src.topics.preprocessing import clean_topic_text, prepare_topic_corpus, tokenize_topic_text


def test_clean_topic_text_normalizes_urls_and_punctuation() -> None:
    cleaned = clean_topic_text("  Battery!!! works at https://example.com  ")
    assert cleaned == "battery works at"
    assert tokenize_topic_text(cleaned) == ["battery", "works", "at"]


def test_prepare_topic_corpus_removes_empty_and_duplicate_reviews() -> None:
    dataframe = pd.DataFrame(
        {
            "review_text": ["Great battery", "great battery", "   ", "Fast shipping"],
            "sentiment_label": ["Positive", "Positive", "Neutral", "Positive"],
        }
    )
    corpus = prepare_topic_corpus(dataframe)
    assert corpus.empty_rows_removed == 1
    assert corpus.duplicate_rows_removed == 1
    assert corpus.texts == ["great battery", "fast shipping"]
    assert len(corpus.dataframe) == 2
