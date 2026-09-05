from __future__ import annotations

import pandas as pd

from src.dashboard.filters import (
    DashboardFilters,
    apply_dashboard_filters,
    get_dashboard_filter_options,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "review_id": "r1",
                "review_text": "Battery life is excellent",
                "date": "2026-01-01",
                "rating": 5,
                "product": "Phone",
                "category": "Hardware",
                "sentiment_label": "Positive",
                "topic_label": "Battery",
                "detected_aspects": ["Battery"],
            },
            {
                "review_id": "r2",
                "review_text": "Delivery was very late",
                "date": "2026-01-15",
                "rating": 1,
                "product": "Phone",
                "category": "Shipping",
                "sentiment_label": "Negative",
                "topic_label": "Delivery",
                "detected_aspects": ["Shipping", "Packaging"],
            },
            {
                "review_id": "r3",
                "review_text": "Support was okay",
                "date": "2026-02-01",
                "rating": 3,
                "product": "Tablet",
                "category": "Service",
                "sentiment_label": "Neutral",
                "topic_label": "Support",
                "detected_aspects": ["Customer Support"],
            },
        ]
    )


def test_filter_options_adapt_to_available_columns() -> None:
    options = get_dashboard_filter_options(_frame())
    assert options.products == ("Phone", "Tablet")
    assert options.sentiments == ("Positive", "Neutral", "Negative")
    assert options.topics == ("Battery", "Delivery", "Support")
    assert options.aspects == ("Battery", "Customer Support", "Packaging", "Shipping")
    assert options.minimum_rating == 1.0
    assert options.maximum_rating == 5.0


def test_apply_dashboard_filters_combines_metadata_and_analysis_filters() -> None:
    filters = DashboardFilters(
        start_date=pd.Timestamp("2026-01-01").date(),
        end_date=pd.Timestamp("2026-01-31").date(),
        products=("Phone",),
        sentiments=("Negative",),
        topics=("Delivery",),
        aspects=("Shipping",),
        search_text="late",
    )
    result = apply_dashboard_filters(_frame(), filters)
    assert result["review_id"].tolist() == ["r2"]


def test_apply_dashboard_filters_date_end_is_inclusive() -> None:
    filters = DashboardFilters(
        start_date=pd.Timestamp("2026-01-15").date(),
        end_date=pd.Timestamp("2026-01-15").date(),
    )
    result = apply_dashboard_filters(_frame(), filters)
    assert result["review_id"].tolist() == ["r2"]


def test_apply_dashboard_filters_does_not_mutate_input() -> None:
    dataframe = _frame()
    original = dataframe.copy(deep=True)
    _ = apply_dashboard_filters(dataframe, DashboardFilters(sentiments=("Positive",)))
    pd.testing.assert_frame_equal(dataframe, original)


def test_filter_round_trip_through_serializable_dict() -> None:
    source = DashboardFilters(
        start_date=pd.Timestamp("2026-01-01").date(),
        products=("Phone",),
        aspects=("Battery",),
        search_text="battery",
    )
    restored = DashboardFilters.from_dict(source.as_dict())
    assert restored == source
    assert restored.is_active()
