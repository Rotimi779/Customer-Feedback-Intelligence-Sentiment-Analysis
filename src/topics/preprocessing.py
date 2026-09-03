"""Text preparation for the NMF topic-modeling stage."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from src.topics.utils import TopicModelError, choose_topic_text_column


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
NON_WORD_PATTERN = re.compile(r"[^\w\s'-]+", flags=re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"(?u)\b[\w][\w'-]+\b")


@dataclass(frozen=True)
class TopicCorpus:
    """Prepared rows and deduplicated texts used to fit the topic model."""

    dataframe: pd.DataFrame
    text_column: str
    texts: list[str]
    empty_rows_removed: int
    duplicate_rows_removed: int


def clean_topic_text(value: object) -> str:
    """Apply conservative topic-specific normalization."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = URL_PATTERN.sub(" ", text)
    text = NON_WORD_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def tokenize_topic_text(value: object) -> list[str]:
    """Tokenize normalized text without adding heavyweight NLP dependencies."""
    return TOKEN_PATTERN.findall(clean_topic_text(value))


def prepare_topic_corpus(
    dataframe: pd.DataFrame,
    *,
    text_column: str | None = None,
) -> TopicCorpus:
    """Remove unusable/duplicate reviews and prepare modeling text.

    Canonical ingestion already removes these rows. Repeating the checks here makes
    the topic module independently testable and safe for labelled public datasets.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    if dataframe.empty:
        raise TopicModelError("Topic modeling requires at least one review.")

    selected = text_column or choose_topic_text_column(dataframe)
    if selected not in dataframe.columns:
        raise TopicModelError(f"Text column '{selected}' does not exist.")

    prepared = dataframe.copy()
    prepared["_topic_text"] = prepared[selected].map(clean_topic_text)

    empty_mask = prepared["_topic_text"].eq("")
    empty_rows_removed = int(empty_mask.sum())
    prepared = prepared.loc[~empty_mask].copy()

    duplicate_mask = prepared["_topic_text"].duplicated(keep="first")
    duplicate_rows_removed = int(duplicate_mask.sum())
    prepared = prepared.loc[~duplicate_mask].copy()

    if prepared.empty:
        raise TopicModelError("No usable unique review text remains for topic modeling.")

    texts = prepared["_topic_text"].tolist()
    prepared = prepared.drop(columns="_topic_text").reset_index(drop=True)

    return TopicCorpus(
        dataframe=prepared,
        text_column=selected,
        texts=texts,
        empty_rows_removed=empty_rows_removed,
        duplicate_rows_removed=duplicate_rows_removed,
    )
