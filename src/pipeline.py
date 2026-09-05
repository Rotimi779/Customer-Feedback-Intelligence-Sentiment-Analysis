"""Application pipeline contracts and completed stage orchestration."""

from __future__ import annotations

import pandas as pd

from src.aspects import AspectAnalysisResult, analyze_aspects
from src.ingestion.cleaning import build_canonical_dataframe
from src.insights import BusinessInsightsResult, generate_business_insights
from src.ingestion.schema import (
    DEFAULT_INGESTION_CONFIG,
    CanonicalizationResult,
    ColumnMapping,
    IngestionConfig,
)
from src.ingestion.validator import DatasetValidationError, validate_dataset
from src.sentiment import SentimentAnalyzer, SentimentInferenceResult, SentimentModelName
from src.topics.evaluation import evaluate_topic_model
from src.topics.modeling import NMFTopicModel, TopicModelResult
from src.topics.utils import TopicModelConfig


def prepare_dataset(
    dataframe: pd.DataFrame,
    mapping: ColumnMapping,
    *,
    config: IngestionConfig = DEFAULT_INGESTION_CONFIG,
) -> CanonicalizationResult:
    """Validate source data and create the canonical ingestion DataFrame."""
    report = validate_dataset(dataframe, mapping, config=config)
    if not report.is_valid:
        raise DatasetValidationError(report)
    return build_canonical_dataframe(dataframe, mapping)


def run_sentiment_stage(
    canonical_df: pd.DataFrame,
    model_name: SentimentModelName | str,
    *,
    analyzer: SentimentAnalyzer | None = None,
) -> SentimentInferenceResult:
    """Run the completed sentiment stage against an existing canonical DataFrame."""
    selected = SentimentModelName(model_name)
    active_analyzer = analyzer or SentimentAnalyzer.load(selected)
    if active_analyzer.model_name is not selected:
        raise ValueError("Provided analyzer does not match the requested sentiment model.")
    return active_analyzer.predict_dataframe(canonical_df)


def run_topic_stage(
    sentiment_df: pd.DataFrame,
    *,
    config: TopicModelConfig | None = None,
    topic_model: NMFTopicModel | None = None,
    stability_runs: int = 3,
) -> TopicModelResult:
    """Discover NMF topics and enrich sentiment results with topic columns."""
    required = {"review_id", "review_text", "clean_text", "sentiment_label", "sentiment_score"}
    missing = sorted(required.difference(sentiment_df.columns))
    if missing:
        raise ValueError(
            "Topic modeling requires sentiment-enriched canonical data. Missing: "
            + ", ".join(missing)
        )

    active_model = topic_model or NMFTopicModel(config or TopicModelConfig())
    if topic_model is not None and config is not None and topic_model.config != config:
        raise ValueError("Provided topic model configuration does not match config.")

    result = active_model.fit_dataframe(sentiment_df)
    evaluation = evaluate_topic_model(
        active_model,
        result.dataframe,
        stability_runs=stability_runs,
    )
    # Keep the result dataclass focused on model outputs while attaching evaluation
    # as transparent metadata for persistence and UI consumers.
    result.model.training_metadata["evaluation"] = evaluation
    return result


def run_aspect_stage(topic_df: pd.DataFrame) -> AspectAnalysisResult:
    """Extract rule-based aspects from Phase 5 topic-enriched review results."""
    required = {
        "review_id",
        "review_text",
        "clean_text",
        "sentiment_label",
        "sentiment_score",
        "topic_id",
        "topic_label",
    }
    missing = sorted(required.difference(topic_df.columns))
    if missing:
        raise ValueError(
            "Aspect analysis requires Phase 5 topic-enriched sentiment data. Missing: "
            + ", ".join(missing)
        )
    return analyze_aspects(topic_df)


def run_insight_stage(
    aspect_df: pd.DataFrame,
    *,
    topic_summary: pd.DataFrame | None = None,
    aspect_summary: pd.DataFrame | None = None,
    aspect_mentions: pd.DataFrame | None = None,
) -> BusinessInsightsResult:
    """Generate Phase 7 business insights from Phase 6 enriched results."""
    required = {
        "review_id",
        "review_text",
        "sentiment_label",
        "topic_id",
        "topic_label",
        "detected_aspects",
        "aspect_sentiment",
    }
    missing = sorted(required.difference(aspect_df.columns))
    if missing:
        raise ValueError(
            "Business insights require Phase 6 aspect-enriched data. Missing: "
            + ", ".join(missing)
        )
    return generate_business_insights(
        aspect_df,
        topic_summary=topic_summary,
        aspect_summary=aspect_summary,
        aspect_mentions=aspect_mentions,
    )


def run_pipeline(
    dataframe: pd.DataFrame,
    *,
    sentiment_model: SentimentModelName | str,
    topic_config: TopicModelConfig | None = None,
    analyzer: SentimentAnalyzer | None = None,
    stability_runs: int = 3,
) -> pd.DataFrame:
    """Run the complete saved-analysis pipeline on a canonical DataFrame.

    The Streamlit dashboard uses a richer orchestration wrapper so it can persist
    intermediate outputs and timings, while this function exposes the final enriched
    review table for programmatic callers.
    """
    sentiment = run_sentiment_stage(dataframe, sentiment_model, analyzer=analyzer)
    topics = run_topic_stage(
        sentiment.dataframe,
        config=topic_config or TopicModelConfig(),
        stability_runs=stability_runs,
    )
    aspects = run_aspect_stage(topics.dataframe)
    insights = run_insight_stage(
        aspects.dataframe,
        topic_summary=topics.summary,
        aspect_summary=aspects.summary,
        aspect_mentions=aspects.mentions,
    )
    return insights.dataframe
