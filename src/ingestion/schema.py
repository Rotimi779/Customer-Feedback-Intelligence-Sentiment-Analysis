"""Data contracts and configuration for CSV ingestion.

The ingestion package converts user-provided CSV files into one stable schema
that every later analysis phase can depend on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final

import pandas as pd


REQUIRED_CANONICAL_COLUMNS: Final[tuple[str, ...]] = (
    "review_id",
    "review_text",
    "clean_text",
)
OPTIONAL_CANONICAL_COLUMNS: Final[tuple[str, ...]] = (
    "date",
    "rating",
    "product",
    "category",
    "version",
)

TEXT_COLUMN_KEYWORDS: Final[tuple[str, ...]] = (
    "review",
    "feedback",
    "comment",
    "text",
    "content",
    "description",
    "message",
    "body",
)

METADATA_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "date": (
        "date",
        "review_date",
        "created_at",
        "created_date",
        "submitted_at",
        "timestamp",
    ),
    "rating": (
        "rating",
        "score",
        "stars",
        "star_rating",
        "review_score",
    ),
    "product": (
        "product",
        "product_name",
        "game",
        "app",
        "application",
        "service",
        "item",
    ),
    "category": (
        "category",
        "department",
        "product_category",
        "feedback_category",
    ),
    "version": (
        "version",
        "release",
        "app_version",
        "product_version",
        "build",
    ),
}

REVIEW_ID_ALIASES: Final[tuple[str, ...]] = (
    "review_id",
    "feedback_id",
    "comment_id",
    "response_id",
    "id",
)


@dataclass(frozen=True, slots=True)
class IngestionConfig:
    """Runtime limits and heuristics for the ingestion workflow."""

    max_file_size_bytes: int = 25 * 1024 * 1024
    max_rows: int = 100_000
    detection_sample_rows: int = 5_000
    minimum_text_score: float = 0.35
    minimum_non_empty_ratio: float = 0.05
    preview_rows: int = 10
    supported_encodings: tuple[str, ...] = (
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1",
    )


DEFAULT_INGESTION_CONFIG: Final[IngestionConfig] = IngestionConfig()


@dataclass(frozen=True, slots=True)
class ColumnMapping:
    """Map normalized source columns to the platform's internal schema."""

    text: str
    date: str | None = None
    rating: str | None = None
    product: str | None = None
    category: str | None = None
    version: str | None = None

    def optional_items(self) -> tuple[tuple[str, str | None], ...]:
        """Return optional canonical/source mappings in canonical order."""
        return (
            ("date", self.date),
            ("rating", self.rating),
            ("product", self.product),
            ("category", self.category),
            ("version", self.version),
        )

    def as_dict(self) -> dict[str, str | None]:
        """Return a serializable representation for session state."""
        return {
            "text": self.text,
            "date": self.date,
            "rating": self.rating,
            "product": self.product,
            "category": self.category,
            "version": self.version,
        }


class ValidationSeverity(str, Enum):
    """Severity levels surfaced to the Streamlit interface."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class ValidationMessage:
    """One user-facing validation result."""

    code: str
    message: str
    severity: ValidationSeverity
    remediation: str | None = None


@dataclass(slots=True)
class ValidationReport:
    """Collection of validation messages for one dataset."""

    messages: list[ValidationMessage] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return whether no blocking validation errors were recorded."""
        return not any(
            item.severity is ValidationSeverity.ERROR for item in self.messages
        )

    @property
    def errors(self) -> tuple[ValidationMessage, ...]:
        """Return only blocking messages."""
        return tuple(
            item
            for item in self.messages
            if item.severity is ValidationSeverity.ERROR
        )

    @property
    def warnings(self) -> tuple[ValidationMessage, ...]:
        """Return only non-blocking warnings."""
        return tuple(
            item
            for item in self.messages
            if item.severity is ValidationSeverity.WARNING
        )

    def extend(self, other: "ValidationReport") -> None:
        """Append messages from another report."""
        self.messages.extend(other.messages)

    def add(
        self,
        *,
        code: str,
        message: str,
        severity: ValidationSeverity,
        remediation: str | None = None,
    ) -> None:
        """Append a validation message."""
        self.messages.append(
            ValidationMessage(
                code=code,
                message=message,
                severity=severity,
                remediation=remediation,
            )
        )


@dataclass(frozen=True, slots=True)
class IngestionStatistics:
    """Row-level changes made while creating the canonical dataset."""

    input_rows: int
    output_rows: int
    empty_reviews_removed: int
    duplicate_reviews_removed: int


@dataclass(frozen=True, slots=True)
class CanonicalizationResult:
    """Canonical dataset plus auditable cleaning statistics."""

    dataframe: pd.DataFrame
    statistics: IngestionStatistics
    warnings: tuple[ValidationMessage, ...] = ()
