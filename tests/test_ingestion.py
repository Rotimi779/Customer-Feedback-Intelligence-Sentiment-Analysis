"""Unit tests for loading, detection, validation, and canonicalization."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

from src.ingestion import (
    ColumnMapping,
    EmptyCSVError,
    FileSizeLimitError,
    IngestionConfig,
    RowLimitError,
    UnsupportedFileTypeError,
    build_canonical_dataframe,
    detect_metadata_columns,
    detect_text_column,
    load_csv,
    normalize_column_names,
    rank_text_columns,
    validate_dataset,
)


def test_load_csv_from_bytes() -> None:
    loaded = load_csv(b"review,rating\nGreat app,5\nSlow update,2\n", filename="reviews.csv")

    assert loaded.filename == "reviews.csv"
    assert loaded.encoding == "utf-8-sig"
    assert loaded.dataframe.shape == (2, 2)


def test_load_csv_uses_encoding_fallback() -> None:
    payload = "review\nTrès bon café\n".encode("cp1252")

    loaded = load_csv(payload, filename="reviews.csv")

    assert loaded.encoding == "cp1252"
    assert loaded.dataframe.loc[0, "review"] == "Très bon café"


def test_load_csv_rejects_non_csv_extension() -> None:
    with pytest.raises(UnsupportedFileTypeError, match="Only CSV"):
        load_csv(b"review\nGood\n", filename="reviews.txt")


def test_load_csv_rejects_empty_payload() -> None:
    with pytest.raises(EmptyCSVError, match="empty"):
        load_csv(b"", filename="reviews.csv")


def test_load_csv_enforces_file_size_limit() -> None:
    config = IngestionConfig(max_file_size_bytes=8)

    with pytest.raises(FileSizeLimitError, match="larger"):
        load_csv(b"review\nThis is too large\n", filename="reviews.csv", config=config)


def test_load_csv_enforces_row_limit_without_reading_every_row() -> None:
    config = IngestionConfig(max_rows=2)

    with pytest.raises(RowLimitError, match="row limit"):
        load_csv(
            b"review\nFirst review\nSecond review\nThird review\n",
            filename="reviews.csv",
            config=config,
        )


def test_loader_accepts_binary_file_object() -> None:
    buffer = BytesIO(b"review\nA useful customer comment\n")
    buffer.name = "feedback.csv"  # type: ignore[attr-defined]

    loaded = load_csv(buffer)

    assert len(loaded.dataframe) == 1


def test_column_names_are_normalized_and_deduplicated() -> None:
    dataframe = pd.DataFrame(
        [["A", "B", 5]],
        columns=["Review Text", "Review-Text", "Star Rating"],
    )

    normalized, mapping = normalize_column_names(dataframe)

    assert list(normalized.columns) == ["review_text", "review_text_2", "star_rating"]
    assert mapping["Review Text"] == "review_text"


def test_text_column_detection_prefers_review_content() -> None:
    dataframe = pd.DataFrame(
        {
            "product": ["App", "App", "Web"],
            "customer_comments": [
                "The navigation is clear and easy to use.",
                "The latest release crashes on startup.",
                "Support solved my problem quickly.",
            ],
        }
    )

    assert detect_text_column(dataframe) == "customer_comments"
    assert rank_text_columns(dataframe)[0].column == "customer_comments"


def test_detection_returns_none_without_usable_text() -> None:
    dataframe = pd.DataFrame({"id": [1, 2], "rating": [4, 5]})

    assert detect_text_column(dataframe) is None


def test_optional_metadata_detection() -> None:
    dataframe = pd.DataFrame(
        {
            "review_text": ["The product works well", "The app is slow"],
            "review_date": ["2026-01-01", "2026-01-02"],
            "stars": [5, 2],
            "app": ["Mobile", "Mobile"],
            "release": ["1.0", "1.1"],
        }
    )

    detected = detect_metadata_columns(dataframe, text_column="review_text")

    assert detected == {
        "date": "review_date",
        "rating": "stars",
        "product": "app",
        "version": "release",
    }


def test_validation_reports_empty_and_duplicate_reviews() -> None:
    dataframe = pd.DataFrame(
        {
            "review": ["Great product", None, "  great   product  "],
        }
    )

    report = validate_dataset(dataframe, ColumnMapping(text="review"))
    codes = {message.code for message in report.messages}

    assert report.is_valid
    assert "empty_reviews_removed" in codes
    assert "duplicate_reviews_removed" in codes


def test_validation_rejects_non_text_manual_selection() -> None:
    dataframe = pd.DataFrame(
        {
            "rating": [1, 2, 3],
            "review": ["Bad experience", "Average experience", "Great experience"],
        }
    )

    report = validate_dataset(dataframe, ColumnMapping(text="rating"))

    assert not report.is_valid
    assert {message.code for message in report.errors} == {
        "text_column_not_text_like"
    }


def test_canonical_dataframe_removes_empty_and_duplicate_reviews() -> None:
    dataframe = pd.DataFrame(
        {
            "review_id": ["A", "B", "C", "D"],
            "review": ["  Great   app  ", None, "great app", "Support was helpful"],
            "created_at": ["2026-01-01", "2026-01-02", "bad-date", "2026-01-04"],
            "stars": [5, 1, 5, 4],
            "product_name": ["Mobile", "Mobile", "Mobile", "Support"],
        }
    )
    mapping = ColumnMapping(
        text="review",
        date="created_at",
        rating="stars",
        product="product_name",
    )

    result = build_canonical_dataframe(dataframe, mapping)

    assert list(result.dataframe.columns) == [
        "review_id",
        "review_text",
        "clean_text",
        "date",
        "rating",
        "product",
    ]
    assert result.dataframe["review_id"].tolist() == ["A", "D"]
    assert result.dataframe.loc[0, "review_text"] == "  Great   app  "
    assert result.dataframe.loc[0, "clean_text"] == "Great app"
    assert result.statistics.input_rows == 4
    assert result.statistics.output_rows == 2
    assert result.statistics.empty_reviews_removed == 1
    assert result.statistics.duplicate_reviews_removed == 1


def test_generated_review_ids_are_stable_and_unique() -> None:
    dataframe = pd.DataFrame(
        {"comment": ["First detailed review", "Second detailed review"]}
    )

    result = build_canonical_dataframe(dataframe, ColumnMapping(text="comment"))

    assert result.dataframe["review_id"].tolist() == [
        "review_000001",
        "review_000002",
    ]
