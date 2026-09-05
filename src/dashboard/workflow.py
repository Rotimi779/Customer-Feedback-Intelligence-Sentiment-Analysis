"""End-to-end orchestration for the integrated Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Any, Callable, MutableMapping

import pandas as pd

from src.aspects import AspectAnalysisResult
from src.insights import BusinessInsightsResult
from src.pipeline import run_aspect_stage, run_insight_stage, run_sentiment_stage, run_topic_stage
from src.sentiment import SentimentAnalyzer, SentimentInferenceResult, SentimentModelName
from src.topics.modeling import TopicModelResult
from src.topics.utils import TopicModelConfig

StageCallback = Callable[[str, str], None]


@dataclass(frozen=True)
class FullAnalysisResult:
    """All stage outputs produced by one full dashboard analysis run."""

    sentiment: SentimentInferenceResult
    topics: TopicModelResult
    aspects: AspectAnalysisResult
    insights: BusinessInsightsResult
    timings: dict[str, float]

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.insights.dataframe


def _notify(callback: StageCallback | None, stage: str, message: str) -> None:
    if callback is not None:
        callback(stage, message)


def run_full_analysis(
    canonical_df: pd.DataFrame,
    *,
    sentiment_model: SentimentModelName | str,
    topic_config: TopicModelConfig | None = None,
    sentiment_analyzer: SentimentAnalyzer | None = None,
    stability_runs: int = 3,
    progress_callback: StageCallback | None = None,
    sentiment_stage: Callable[..., SentimentInferenceResult] = run_sentiment_stage,
    topic_stage: Callable[..., TopicModelResult] = run_topic_stage,
    aspect_stage: Callable[..., AspectAnalysisResult] = run_aspect_stage,
    insight_stage: Callable[..., BusinessInsightsResult] = run_insight_stage,
) -> FullAnalysisResult:
    """Run sentiment → topics → aspects → insights without touching Streamlit state."""
    if not isinstance(canonical_df, pd.DataFrame) or canonical_df.empty:
        raise ValueError("Full analysis requires a non-empty canonical DataFrame.")

    selected = SentimentModelName(sentiment_model)
    config = topic_config or TopicModelConfig()
    timings: dict[str, float] = {}

    _notify(progress_callback, "sentiment", "Running sentiment analysis")
    start = time.perf_counter()
    sentiment = sentiment_stage(canonical_df, selected, analyzer=sentiment_analyzer)
    timings["sentiment_seconds"] = time.perf_counter() - start

    _notify(progress_callback, "topics", "Discovering topics")
    start = time.perf_counter()
    topics = topic_stage(sentiment.dataframe, config=config, stability_runs=stability_runs)
    timings["topic_seconds"] = time.perf_counter() - start

    _notify(progress_callback, "aspects", "Extracting aspects")
    start = time.perf_counter()
    aspects = aspect_stage(topics.dataframe)
    timings["aspect_seconds"] = time.perf_counter() - start

    _notify(progress_callback, "insights", "Generating business insights")
    start = time.perf_counter()
    insights = insight_stage(
        aspects.dataframe,
        topic_summary=topics.summary,
        aspect_summary=aspects.summary,
        aspect_mentions=aspects.mentions,
    )
    timings["insight_seconds"] = time.perf_counter() - start
    timings["total_seconds"] = sum(timings.values())
    _notify(progress_callback, "complete", "Analysis complete")

    return FullAnalysisResult(
        sentiment=sentiment,
        topics=topics,
        aspects=aspects,
        insights=insights,
        timings=timings,
    )


def persist_full_analysis(
    state: MutableMapping[str, Any],
    result: FullAnalysisResult,
    *,
    source_signature: str,
) -> None:
    """Persist a complete run into the existing session-state contract."""
    state["results_df"] = result.dataframe.copy()
    state["sentiment_complete"] = True
    state["selected_sentiment_model"] = result.sentiment.model_name.value
    state["sentiment_runtime_seconds"] = result.timings.get("sentiment_seconds")
    state["sentiment_source_signature"] = source_signature

    state["topic_summary"] = result.topics.summary
    state["topic_complete"] = True
    state["topic_metrics"] = result.topics.model.training_metadata.get("evaluation")
    state["topic_source_signature"] = f"{source_signature}:{result.sentiment.model_name.value}"
    state["topic_config"] = result.topics.model.config.as_dict()
    state["topic_model_runtime"] = result.timings.get("topic_seconds")
    state["topic_representatives"] = result.topics.representative_review_ids

    state["aspect_summary"] = result.aspects.summary
    state["aspect_mentions"] = result.aspects.mentions
    state["aspect_metrics"] = result.aspects.evaluation
    state["aspect_complete"] = True
    aspect_parts = [
        source_signature,
        result.sentiment.model_name.value,
        str(state["topic_source_signature"]),
        str(state["topic_config"]),
    ]
    state["aspect_source_signature"] = hashlib.sha256(
        "|".join(aspect_parts).encode("utf-8")
    ).hexdigest()
    state["aspect_runtime_seconds"] = result.timings.get("aspect_seconds")

    state["insights"] = result.insights
    state["insight_complete"] = True
    insight_parts = [
        source_signature,
        result.sentiment.model_name.value,
        str(state["topic_source_signature"]),
        str(state["topic_config"]),
        str(state["aspect_source_signature"]),
    ]
    state["insight_source_signature"] = hashlib.sha256(
        "|".join(insight_parts).encode("utf-8")
    ).hexdigest()
    state["insight_runtime_seconds"] = result.timings.get("insight_seconds")
    state["analysis_complete"] = True
    state["analysis_running"] = False
    state["analysis_stage"] = "complete"
    state["last_error"] = None
    state["filtered_df"] = None
    state["active_filters"] = {}
