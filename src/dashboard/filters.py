"""Reusable non-mutating dashboard filters shared across analysis pages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DashboardFilters:
    """Filter selections supported by the integrated dashboard."""

    start_date: date | datetime | pd.Timestamp | None = None
    end_date: date | datetime | pd.Timestamp | None = None
    products: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    rating_min: float | None = None
    rating_max: float | None = None
    sentiments: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    aspects: tuple[str, ...] = ()
    search_text: str = ""

    def is_active(self) -> bool:
        return any(
            (
                self.start_date is not None,
                self.end_date is not None,
                bool(self.products),
                bool(self.categories),
                self.rating_min is not None,
                self.rating_max is not None,
                bool(self.sentiments),
                bool(self.topics),
                bool(self.aspects),
                bool(self.search_text.strip()),
            )
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("start_date", "end_date"):
            value = payload[key]
            payload[key] = value.isoformat() if value is not None else None
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "DashboardFilters":
        if not payload:
            return cls()
        data = dict(payload)
        for key in ("products", "categories", "sentiments", "topics", "aspects"):
            data[key] = tuple(data.get(key) or ())
        for key in ("start_date", "end_date"):
            value = data.get(key)
            if isinstance(value, str) and value:
                data[key] = pd.Timestamp(value).date()
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in allowed})


@dataclass(frozen=True)
class DashboardFilterOptions:
    minimum_date: pd.Timestamp | None
    maximum_date: pd.Timestamp | None
    products: tuple[str, ...]
    categories: tuple[str, ...]
    minimum_rating: float | None
    maximum_rating: float | None
    sentiments: tuple[str, ...]
    topics: tuple[str, ...]
    aspects: tuple[str, ...]


def _string_options(dataframe: pd.DataFrame, column: str) -> tuple[str, ...]:
    if column not in dataframe.columns:
        return ()
    values = dataframe[column].dropna().astype(str).str.strip()
    return tuple(sorted(value for value in values.unique() if value))


def _aspect_options(dataframe: pd.DataFrame) -> tuple[str, ...]:
    if "aspect" in dataframe.columns:
        return _string_options(dataframe, "aspect")
    if "detected_aspects" not in dataframe.columns:
        return ()
    values: set[str] = set()
    for item in dataframe["detected_aspects"]:
        if isinstance(item, (list, tuple, set)):
            values.update(str(value) for value in item if str(value).strip())
    return tuple(sorted(values))


def get_dashboard_filter_options(dataframe: pd.DataFrame) -> DashboardFilterOptions:
    """Inspect available columns and return only supported filter values."""
    minimum_date = maximum_date = None
    if "date" in dataframe.columns:
        dates = pd.to_datetime(dataframe["date"], errors="coerce").dropna()
        if not dates.empty:
            minimum_date = pd.Timestamp(dates.min())
            maximum_date = pd.Timestamp(dates.max())

    minimum_rating = maximum_rating = None
    if "rating" in dataframe.columns:
        ratings = pd.to_numeric(dataframe["rating"], errors="coerce").dropna()
        if not ratings.empty:
            minimum_rating = float(ratings.min())
            maximum_rating = float(ratings.max())

    sentiment_order = ("Positive", "Neutral", "Negative")
    observed_sentiments = set(_string_options(dataframe, "sentiment_label"))
    sentiments = tuple(label for label in sentiment_order if label in observed_sentiments)
    sentiments += tuple(sorted(observed_sentiments.difference(sentiment_order)))

    return DashboardFilterOptions(
        minimum_date=minimum_date,
        maximum_date=maximum_date,
        products=_string_options(dataframe, "product"),
        categories=_string_options(dataframe, "category"),
        minimum_rating=minimum_rating,
        maximum_rating=maximum_rating,
        sentiments=sentiments,
        topics=_string_options(dataframe, "topic_label"),
        aspects=_aspect_options(dataframe),
    )


def _contains_selected_aspect(value: object, selected: set[str]) -> bool:
    if isinstance(value, (list, tuple, set)):
        return bool(selected.intersection(str(item) for item in value))
    return str(value) in selected


def apply_dashboard_filters(
    dataframe: pd.DataFrame,
    filters: DashboardFilters | None = None,
) -> pd.DataFrame:
    """Return a filtered copy without mutating the saved analysis DataFrame."""
    active = filters or DashboardFilters()
    result = dataframe.copy()

    if "date" in result.columns and (active.start_date is not None or active.end_date is not None):
        dates = pd.to_datetime(result["date"], errors="coerce")
        mask = dates.notna()
        if active.start_date is not None:
            mask &= dates >= pd.Timestamp(active.start_date)
        if active.end_date is not None:
            # Include the whole selected calendar day.
            mask &= dates < pd.Timestamp(active.end_date) + pd.Timedelta(days=1)
        result = result.loc[mask]

    if active.products and "product" in result.columns:
        result = result.loc[result["product"].astype(str).isin(active.products)]
    if active.categories and "category" in result.columns:
        result = result.loc[result["category"].astype(str).isin(active.categories)]

    if "rating" in result.columns and (active.rating_min is not None or active.rating_max is not None):
        ratings = pd.to_numeric(result["rating"], errors="coerce")
        mask = ratings.notna()
        if active.rating_min is not None:
            mask &= ratings >= active.rating_min
        if active.rating_max is not None:
            mask &= ratings <= active.rating_max
        result = result.loc[mask]

    if active.sentiments and "sentiment_label" in result.columns:
        result = result.loc[result["sentiment_label"].astype(str).isin(active.sentiments)]
    if active.topics and "topic_label" in result.columns:
        result = result.loc[result["topic_label"].astype(str).isin(active.topics)]

    if active.aspects:
        selected = set(active.aspects)
        if "aspect" in result.columns:
            result = result.loc[result["aspect"].astype(str).isin(selected)]
        elif "detected_aspects" in result.columns:
            result = result.loc[result["detected_aspects"].apply(
                lambda value: _contains_selected_aspect(value, selected)
            )]

    search = active.search_text.strip()
    if search and "review_text" in result.columns:
        result = result.loc[
            result["review_text"].fillna("").astype(str).str.contains(
                search, case=False, regex=False, na=False
            )
        ]

    return result.copy()
