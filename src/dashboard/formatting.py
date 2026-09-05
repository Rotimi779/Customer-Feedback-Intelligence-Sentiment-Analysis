"""Consistent dashboard labels, colors, and formatting helpers."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

SENTIMENT_COLORS: dict[str, str] = {
    "Positive": "#2E8B57",
    "Neutral": "#808080",
    "Negative": "#D9534F",
}
SENTIMENT_ORDER: tuple[str, ...] = ("Positive", "Neutral", "Negative")


def format_percentage(value: float | int | None, *, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{decimals}%}"


def format_confidence(value: float | int | None) -> str:
    return format_percentage(value)


def format_date(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    parsed = pd.Timestamp(value)
    return parsed.strftime("%Y-%m-%d")


def format_large_number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    number = float(value)
    magnitude = abs(number)
    if magnitude >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    if magnitude >= 1_000:
        return f"{number / 1_000:.1f}K"
    return f"{number:,.0f}"


def format_label(value: object) -> str:
    return str(value).replace("_", " ").strip().title()


def truncate_text(value: object, *, max_length: int = 180) -> str:
    text = " ".join(str(value).split())
    if len(text) <= max_length:
        return text
    if max_length <= 1:
        return "…"[:max_length]
    return text[: max_length - 1].rstrip() + "…"
