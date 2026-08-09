"""Integration tests from uploaded CSV bytes to the canonical data contract."""

from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion import (
    ColumnMapping,
    DatasetValidationError,
    detect_metadata_columns,
    detect_text_column,
    load_csv,
    normalize_column_names,
)
from src.pipeline import prepare_dataset


def test_valid_upload_to_canonical_dataframe() -> None:
    payload = (
        b"Feedback,Review Date,Stars,Product\n"
        b"The interface is easy to use,2026-01-01,5,Mobile App\n"
        b"The latest update is very slow,2026-01-02,2,Mobile App\n"
    )

    loaded = load_csv(payload, filename="feedback.csv")
    normalized, _ = normalize_column_names(loaded.dataframe)
    text_column = detect_text_column(normalized)
    assert text_column is not None
    metadata = detect_metadata_columns(normalized, text_column=text_column)

    result = prepare_dataset(
        normalized,
        ColumnMapping(
            text=text_column,
            date=metadata.get("date"),
            rating=metadata.get("rating"),
            product=metadata.get("product"),
        ),
    )

    assert list(result.dataframe.columns) == [
        "review_id",
        "review_text",
        "clean_text",
        "date",
        "rating",
        "product",
    ]
    assert len(result.dataframe) == 2
    assert pd.api.types.is_datetime64_any_dtype(result.dataframe["date"])


def test_text_only_upload_is_valid_without_optional_metadata() -> None:
    payload = b"message\nCustomer support was fast and helpful\nThe checkout page failed twice\n"

    loaded = load_csv(payload, filename="messages.csv")
    normalized, _ = normalize_column_names(loaded.dataframe)
    text_column = detect_text_column(normalized)
    assert text_column == "message"

    result = prepare_dataset(normalized, ColumnMapping(text=text_column))

    assert list(result.dataframe.columns) == [
        "review_id",
        "review_text",
        "clean_text",
    ]
    assert len(result.dataframe) == 2


def test_invalid_mapping_never_reaches_canonicalization() -> None:
    dataframe = pd.DataFrame(
        {
            "rating": [1, 2, 3],
            "product": ["A", "B", "C"],
        }
    )

    with pytest.raises(DatasetValidationError):
        prepare_dataset(dataframe, ColumnMapping(text="rating"))


def test_duplicate_optional_mapping_is_rejected() -> None:
    dataframe = pd.DataFrame(
        {
            "review": ["A detailed review of the application"],
            "created_at": ["2026-01-01"],
        }
    )

    with pytest.raises(DatasetValidationError, match="mapped to both"):
        prepare_dataset(
            dataframe,
            ColumnMapping(
                text="review",
                date="created_at",
                product="created_at",
            ),
        )
