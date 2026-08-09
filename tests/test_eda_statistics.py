"""Unit tests for EDA descriptive statistics."""

from __future__ import annotations

import pandas as pd

from src.eda.statistics import (
    add_review_length_columns,
    calculate_common_words,
    calculate_rating_statistics,
    calculate_review_length_statistics,
    infer_time_frequency,
    summarize_reviews_over_time,
)


def test_review_length_columns_do_not_mutate_input() -> None:
    dataframe = pd.DataFrame({"review_text": ["One two", "Three words here"]})
    original_columns = list(dataframe.columns)

    enriched = add_review_length_columns(dataframe)

    assert list(dataframe.columns) == original_columns
    assert enriched["review_word_count"].tolist() == [2, 3]
    assert enriched["review_character_count"].tolist() == [7, 16]


def test_review_length_statistics() -> None:
    dataframe = pd.DataFrame(
        {"review_text": ["one two", "one two three four", "single"]}
    )

    statistics = calculate_review_length_statistics(dataframe)

    assert statistics.average_words == 7 / 3
    assert statistics.median_words == 2.0
    assert statistics.minimum_words == 1
    assert statistics.maximum_words == 4


def test_rating_statistics_ignore_invalid_values() -> None:
    dataframe = pd.DataFrame({"rating": [5, "4", "invalid", None]})

    statistics = calculate_rating_statistics(dataframe)

    assert statistics is not None
    assert statistics.count == 2
    assert statistics.average == 4.5
    assert statistics.minimum == 4.0
    assert statistics.maximum == 5.0


def test_common_words_apply_light_preprocessing() -> None:
    dataframe = pd.DataFrame(
        {
            "clean_text": [
                "The app is fast and helpful",
                "Helpful support made the app fast",
            ]
        }
    )

    terms = calculate_common_words(dataframe, top_n=3)

    assert terms["term"].tolist() == ["app", "fast", "helpful"]
    assert terms["count"].tolist() == [2, 2, 2]
    assert "the" not in terms["term"].tolist()


def test_reviews_over_time_aggregates_dates() -> None:
    dataframe = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-01", "2026-01-03", "bad"],
        }
    )

    timeline = summarize_reviews_over_time(dataframe)

    assert timeline["review_count"].tolist() == [2, 1]
    assert timeline["period"].dt.date.astype(str).tolist() == [
        "2026-01-01",
        "2026-01-03",
    ]


def test_time_frequency_adapts_to_date_span() -> None:
    short = pd.DataFrame({"date": ["2026-01-01", "2026-02-01"]})
    medium = pd.DataFrame({"date": ["2025-01-01", "2026-01-01"]})
    long = pd.DataFrame({"date": ["2020-01-01", "2026-01-01"]})

    assert infer_time_frequency(short) == "D"
    assert infer_time_frequency(medium) == "W"
    assert infer_time_frequency(long) == "M"
