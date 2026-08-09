"""Integration tests from ingestion output through EDA utilities."""

from __future__ import annotations

from pathlib import Path

from src.eda import (
    DatasetFilters,
    apply_filters,
    build_dataset_summary,
    calculate_common_words,
    get_filter_options,
)
from src.ingestion import (
    ColumnMapping,
    detect_metadata_columns,
    detect_text_column,
    load_csv,
    normalize_column_names,
)
from src.pipeline import prepare_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_sample_upload_to_filtered_eda_summary() -> None:
    sample_path = PROJECT_ROOT / "data" / "sample" / "sample_customer_reviews.csv"
    loaded = load_csv(sample_path)
    normalized, _ = normalize_column_names(loaded.dataframe)
    text_column = detect_text_column(normalized)
    assert text_column is not None
    metadata = detect_metadata_columns(normalized, text_column=text_column)

    canonical = prepare_dataset(
        normalized,
        ColumnMapping(
            text=text_column,
            date=metadata.get("date"),
            rating=metadata.get("rating"),
            product=metadata.get("product"),
        ),
    ).dataframe

    options = get_filter_options(canonical)
    assert options.minimum_date is not None
    assert options.maximum_rating == 5.0

    filtered = apply_filters(
        canonical,
        DatasetFilters(rating_min=4.0),
    )
    summary = build_dataset_summary(filtered)
    terms = calculate_common_words(filtered, top_n=10)

    assert summary.total_reviews == 5
    assert summary.average_rating == 4.8
    assert not terms.empty


def test_text_only_dataset_supports_complete_eda_flow() -> None:
    payload = (
        b"message\n"
        b"Customer support was fast and helpful\n"
        b"The checkout page failed twice\n"
    )
    loaded = load_csv(payload, filename="messages.csv")
    normalized, _ = normalize_column_names(loaded.dataframe)
    text_column = detect_text_column(normalized)
    assert text_column is not None

    canonical = prepare_dataset(
        normalized,
        ColumnMapping(text=text_column),
    ).dataframe
    summary = build_dataset_summary(canonical)

    assert summary.total_reviews == 2
    assert summary.average_rating is None
    assert summary.minimum_date is None
