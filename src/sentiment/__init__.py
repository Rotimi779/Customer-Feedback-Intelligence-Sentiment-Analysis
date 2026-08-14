"""Public sentiment-analysis API."""

from src.sentiment.baseline import (
    BaselineModelNotReadyError,
    BaselineSentimentModel,
)
from src.sentiment.evaluation import (
    build_model_comparison,
    load_production_model_selection,
    load_saved_model_comparison,
    save_production_model_selection,
)
from src.sentiment.inference import (
    SentimentAnalyzer,
    SentimentInferenceResult,
    SentimentModelName,
    available_sentiment_models,
    baseline_model_available,
    transformer_model_available,
)
from src.sentiment.metrics import compute_classification_metrics
from src.sentiment.preprocessing import (
    prepare_classical_text,
    prepare_classical_texts,
    prepare_transformer_text,
    prepare_transformer_texts,
)
from src.sentiment.transformer import (
    DistilBertSentimentModel,
    TransformerDependencyError,
    TransformerModelNotReadyError,
)

__all__ = [
    "BaselineModelNotReadyError",
    "BaselineSentimentModel",
    "DistilBertSentimentModel",
    "SentimentAnalyzer",
    "SentimentInferenceResult",
    "SentimentModelName",
    "TransformerDependencyError",
    "TransformerModelNotReadyError",
    "available_sentiment_models",
    "baseline_model_available",
    "build_model_comparison",
    "compute_classification_metrics",
    "load_production_model_selection",
    "load_saved_model_comparison",
    "prepare_classical_text",
    "prepare_classical_texts",
    "prepare_transformer_text",
    "prepare_transformer_texts",
    "save_production_model_selection",
    "transformer_model_available",
]
