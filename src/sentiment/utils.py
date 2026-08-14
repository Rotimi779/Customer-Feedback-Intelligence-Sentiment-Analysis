"""Shared sentiment configuration, data preparation, and artifact helpers."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SENTIMENT_LABELS: tuple[str, ...] = ("Negative", "Neutral", "Positive")
LABEL_TO_ID: dict[str, int] = {label: index for index, label in enumerate(SENTIMENT_LABELS)}
ID_TO_LABEL: dict[int, str] = {index: label for label, index in LABEL_TO_ID.items()}
DEFAULT_RANDOM_STATE = 42
DEFAULT_DISTILBERT_MODEL = "distilbert/distilbert-base-uncased"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_DIR = PROJECT_ROOT / "models" / "logistic_regression"
DEFAULT_DISTILBERT_DIR = PROJECT_ROOT / "models" / "distilbert"
PRODUCTION_SELECTION_PATH = PROJECT_ROOT / "models" / "production_model.json"


class SentimentDataError(ValueError):
    """Raised when labelled training/evaluation data does not meet the contract."""


@dataclass(frozen=True)
class SentimentSplitConfig:
    """Reproducible split proportions required by the experiment plan."""

    train_size: float = 0.70
    validation_size: float = 0.15
    test_size: float = 0.15
    random_state: int = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        total = self.train_size + self.validation_size + self.test_size
        if not np.isclose(total, 1.0):
            raise ValueError("Sentiment split proportions must sum to 1.0.")
        if min(self.train_size, self.validation_size, self.test_size) <= 0:
            raise ValueError("All split proportions must be greater than zero.")


@dataclass(frozen=True)
class BaselineTrainingConfig:
    """Configuration for the TF-IDF + Logistic Regression baseline."""

    ngram_min: int = 1
    ngram_max: int = 2
    max_features: int = 20_000
    regularization_c: float = 1.0
    max_iterations: int = 1_000
    class_weight: str | None = "balanced"
    random_state: int = DEFAULT_RANDOM_STATE


@dataclass(frozen=True)
class TransformerTrainingConfig:
    """Configuration for three-class DistilBERT fine-tuning."""

    base_model_name: str = DEFAULT_DISTILBERT_MODEL
    max_length: int = 256
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 4
    early_stopping_patience: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    random_state: int = DEFAULT_RANDOM_STATE


def set_global_seed(seed: int = DEFAULT_RANDOM_STATE) -> None:
    """Seed Python, NumPy, and PyTorch when available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_sentiment_label(value: object, mapping: dict[str, str] | None = None) -> str:
    """Convert one label to the canonical Positive/Neutral/Negative contract.

    ``mapping`` is explicit by design; numeric or domain-specific labels are not
    guessed automatically because an incorrect mapping would invalidate model
    evaluation.
    """
    if pd.isna(value):
        raise SentimentDataError("Sentiment labels cannot be missing.")

    raw = str(value).strip()
    key = raw.casefold()
    if mapping:
        normalized_mapping = {str(k).strip().casefold(): str(v).strip() for k, v in mapping.items()}
        raw = normalized_mapping.get(key, raw)
        key = raw.casefold()

    canonical = {label.casefold(): label for label in SENTIMENT_LABELS}
    if key not in canonical:
        raise SentimentDataError(
            f"Unsupported sentiment label '{value}'. Expected Positive, Neutral, "
            "or Negative, or provide an explicit label mapping."
        )
    return canonical[key]


def prepare_labeled_dataframe(
    dataframe: pd.DataFrame,
    *,
    text_column: str = "review_text",
    label_column: str = "sentiment_label",
    label_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Validate, deduplicate, and standardize a labelled sentiment dataset."""
    missing = [column for column in (text_column, label_column) if column not in dataframe.columns]
    if missing:
        raise SentimentDataError(f"Missing required labelled-data columns: {missing}.")

    prepared = dataframe[[text_column, label_column]].copy()
    prepared[text_column] = prepared[text_column].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    prepared = prepared.loc[prepared[text_column].ne("")].copy()
    if prepared.empty:
        raise SentimentDataError("No usable review text remains after removing empty rows.")

    prepared[label_column] = [
        normalize_sentiment_label(value, label_mapping)
        for value in prepared[label_column].tolist()
    ]

    conflicting = (
        prepared.groupby(text_column, dropna=False)[label_column]
        .nunique()
        .loc[lambda counts: counts > 1]
    )
    if not conflicting.empty:
        raise SentimentDataError(
            "Duplicate review text has conflicting sentiment labels. Resolve those "
            "records before training so evaluation is not ambiguous."
        )

    prepared = prepared.drop_duplicates(subset=[text_column], keep="first").reset_index(drop=True)

    observed = set(prepared[label_column].unique())
    missing_labels = set(SENTIMENT_LABELS) - observed
    if missing_labels:
        raise SentimentDataError(
            "A three-class training dataset is required. Missing labels: "
            + ", ".join(sorted(missing_labels))
            + "."
        )

    class_counts = prepared[label_column].value_counts()
    if int(class_counts.min()) < 7:
        raise SentimentDataError(
            "Each sentiment class needs at least seven examples to create reproducible "
            "70/15/15 stratified splits with all three classes represented."
        )
    return prepared


def split_labeled_dataframe(
    dataframe: pd.DataFrame,
    *,
    text_column: str = "review_text",
    label_column: str = "sentiment_label",
    config: SentimentSplitConfig = SentimentSplitConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create deterministic 70/15/15 stratified train/validation/test splits."""
    temporary_size = config.validation_size + config.test_size
    train_df, temporary_df = train_test_split(
        dataframe,
        test_size=temporary_size,
        stratify=dataframe[label_column],
        random_state=config.random_state,
    )
    relative_test_size = config.test_size / temporary_size
    validation_df, test_df = train_test_split(
        temporary_df,
        test_size=relative_test_size,
        stratify=temporary_df[label_column],
        random_state=config.random_state,
    )
    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def dataframe_fingerprint(
    dataframe: pd.DataFrame,
    *,
    text_column: str,
    label_column: str,
) -> str:
    """Return a stable SHA-256 fingerprint for experiment metadata."""
    payload = dataframe[[text_column, label_column]].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_label_mapping_json(raw: str | None) -> dict[str, str] | None:
    """Parse an optional JSON label mapping supplied on the command line."""
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SentimentDataError("--label-map-json must contain a valid JSON object.") from exc
    if not isinstance(value, dict):
        raise SentimentDataError("--label-map-json must contain a JSON object.")
    return {str(key): str(label) for key, label in value.items()}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write pretty, deterministic JSON and create parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def directory_size_bytes(path: Path) -> int:
    """Return the total size of regular files below ``path``."""
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    """Serialize a configuration dataclass."""
    return asdict(value)
