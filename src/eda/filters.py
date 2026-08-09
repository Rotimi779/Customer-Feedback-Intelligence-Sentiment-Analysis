"""Reusable, non-mutating filters for EDA and later dashboard pages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

import pandas as pd


DateLike = date | datetime | pd.Timestamp | str


@dataclass(frozen=True, slots=True)
class DatasetFilters:
    """All supported filters for the canonical customer-feedback dataset."""

    start_date: DateLike | None = None
    end_date: DateLike | None = None
    products: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    rating_min: float | None = None
    rating_max: float | None = None
    keyword: str = ""


@dataclass(frozen=True, slots=True)
class FilterOptions:
    """Available values and bounds used to build filter widgets."""

    minimum_date: pd.Timestamp | None
    maximum_date: pd.Timestamp | None
    products: tuple[str, ...]
    categories: tuple[str, ...]
    minimum_rating: float | None
    maximum_rating: float | None


def _sorted_strings(values: Iterable[object]) -> tuple[str, ...]:
    cleaned = {
        str(value).strip()
        for value in values
        if not pd.isna(value) and str(value).strip()
    }
    return tuple(sorted(cleaned, key=str.casefold))


def get_filter_options(dataframe: pd.DataFrame) -> FilterOptions:
    """Inspect optional metadata and return safe widget options."""
    minimum_date: pd.Timestamp | None = None
    maximum_date: pd.Timestamp | None = None
    if "date" in dataframe.columns:
        dates = pd.to_datetime(dataframe["date"], errors="coerce").dropna()
        if not dates.empty:
            minimum_date = pd.Timestamp(dates.min())
            maximum_date = pd.Timestamp(dates.max())

    minimum_rating: float | None = None
    maximum_rating: float | None = None
    if "rating" in dataframe.columns:
        ratings = pd.to_numeric(dataframe["rating"], errors="coerce").dropna()
        if not ratings.empty:
            minimum_rating = float(ratings.min())
            maximum_rating = float(ratings.max())

    products = (
        _sorted_strings(dataframe["product"].tolist())
        if "product" in dataframe.columns
        else ()
    )
    categories = (
        _sorted_strings(dataframe["category"].tolist())
        if "category" in dataframe.columns
        else ()
    )
    return FilterOptions(
        minimum_date=minimum_date,
        maximum_date=maximum_date,
        products=products,
        categories=categories,
        minimum_rating=minimum_rating,
        maximum_rating=maximum_rating,
    )


def _as_timestamp(value: DateLike | None) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid date filter value: {value!r}")
    return pd.Timestamp(parsed)


def apply_filters(
    dataframe: pd.DataFrame,
    filters: DatasetFilters,
    *,
    text_column: str = "review_text",
) -> pd.DataFrame:
    """Return a filtered copy while gracefully skipping unavailable metadata."""
    mask = pd.Series(True, index=dataframe.index, dtype="bool")

    if "date" in dataframe.columns and (
        filters.start_date is not None or filters.end_date is not None
    ):
        dates = pd.to_datetime(dataframe["date"], errors="coerce")
        start = _as_timestamp(filters.start_date)
        end = _as_timestamp(filters.end_date)
        if start is not None:
            mask &= dates.ge(start)
        if end is not None:
            # A date-only value is inclusive for the full selected calendar day.
            if end == end.normalize():
                end = end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            mask &= dates.le(end)

    if "product" in dataframe.columns and filters.products:
        selected_products = {value.casefold() for value in filters.products}
        normalized_products = (
            dataframe["product"].astype("string").fillna("").str.casefold()
        )
        mask &= normalized_products.isin(selected_products)

    if "category" in dataframe.columns and filters.categories:
        selected_categories = {value.casefold() for value in filters.categories}
        normalized_categories = (
            dataframe["category"].astype("string").fillna("").str.casefold()
        )
        mask &= normalized_categories.isin(selected_categories)

    if "rating" in dataframe.columns and (
        filters.rating_min is not None or filters.rating_max is not None
    ):
        ratings = pd.to_numeric(dataframe["rating"], errors="coerce")
        if filters.rating_min is not None:
            mask &= ratings.ge(filters.rating_min)
        if filters.rating_max is not None:
            mask &= ratings.le(filters.rating_max)

    keyword = filters.keyword.strip()
    if keyword and text_column in dataframe.columns:
        mask &= (
            dataframe[text_column]
            .astype("string")
            .fillna("")
            .str.contains(keyword, case=False, regex=False)
        )

    return dataframe.loc[mask].copy().reset_index(drop=True)
