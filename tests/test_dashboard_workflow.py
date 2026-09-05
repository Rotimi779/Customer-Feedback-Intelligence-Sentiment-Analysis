from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.dashboard.workflow import FullAnalysisResult, persist_full_analysis, run_full_analysis
from src.sentiment import SentimentModelName
from src.topics.utils import TopicModelConfig


def _canonical() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": ["r1", "r2"],
            "review_text": ["great battery", "bad shipping"],
            "clean_text": ["great battery", "bad shipping"],
        }
    )


def test_full_analysis_runs_stages_in_order_without_ui_state() -> None:
    events: list[str] = []

    def sentiment_stage(dataframe, model_name, analyzer=None):
        events.append("sentiment")
        enriched = dataframe.copy()
        enriched["sentiment_label"] = ["Positive", "Negative"]
        enriched["sentiment_score"] = [0.9, 0.9]
        return SimpleNamespace(
            dataframe=enriched,
            model_name=SentimentModelName(model_name),
            mean_confidence=0.9,
        )

    config = TopicModelConfig(n_topics=5)

    def topic_stage(dataframe, config=None, stability_runs=3):
        events.append("topics")
        enriched = dataframe.copy()
        enriched["topic_id"] = [0, 1]
        enriched["topic_label"] = ["Battery", "Shipping"]
        model = SimpleNamespace(
            config=config,
            training_metadata={"evaluation": {"topic_coverage": 1.0}},
        )
        return SimpleNamespace(
            dataframe=enriched,
            summary=pd.DataFrame({"topic_id": [0, 1]}),
            representative_review_ids={0: ["r1"], 1: ["r2"]},
            model=model,
        )

    def aspect_stage(dataframe):
        events.append("aspects")
        enriched = dataframe.copy()
        enriched["detected_aspects"] = [["Battery"], ["Shipping"]]
        enriched["aspect_sentiment"] = [
            {"Battery": "Positive"}, {"Shipping": "Negative"}
        ]
        return SimpleNamespace(
            dataframe=enriched,
            summary=pd.DataFrame({"aspect": ["Battery", "Shipping"]}),
            mentions=pd.DataFrame({"aspect": ["Battery", "Shipping"]}),
            evaluation={"aspect_coverage": 1.0},
        )

    def insight_stage(dataframe, **kwargs):
        events.append("insights")
        return SimpleNamespace(dataframe=dataframe.copy())

    result = run_full_analysis(
        _canonical(),
        sentiment_model="distilbert",
        topic_config=config,
        sentiment_stage=sentiment_stage,
        topic_stage=topic_stage,
        aspect_stage=aspect_stage,
        insight_stage=insight_stage,
        progress_callback=lambda stage, message: events.append(f"progress:{stage}"),
    )

    assert [event for event in events if not event.startswith("progress:")] == [
        "sentiment", "topics", "aspects", "insights"
    ]
    assert set(result.dataframe.columns).issuperset(
        {"sentiment_label", "topic_label", "detected_aspects", "aspect_sentiment"}
    )
    assert result.timings["total_seconds"] >= 0


def test_persist_full_analysis_populates_all_stage_state() -> None:
    sentiment = SimpleNamespace(
        dataframe=_canonical(),
        model_name=SentimentModelName.DISTILBERT,
        mean_confidence=0.9,
    )
    model = SimpleNamespace(
        config=TopicModelConfig(n_topics=5),
        training_metadata={"evaluation": {"topic_coverage": 1.0}},
    )
    topic_df = _canonical().assign(
        sentiment_label=["Positive", "Negative"],
        sentiment_score=[0.9, 0.9],
        topic_id=[0, 1],
        topic_label=["Battery", "Shipping"],
    )
    topics = SimpleNamespace(
        dataframe=topic_df,
        summary=pd.DataFrame({"topic_id": [0, 1]}),
        representative_review_ids={0: ["r1"]},
        model=model,
    )
    aspect_df = topic_df.assign(
        detected_aspects=[["Battery"], ["Shipping"]],
        aspect_sentiment=[{"Battery": "Positive"}, {"Shipping": "Negative"}],
    )
    aspects = SimpleNamespace(
        dataframe=aspect_df,
        summary=pd.DataFrame({"aspect": ["Battery"]}),
        mentions=pd.DataFrame({"aspect": ["Battery"]}),
        evaluation={"aspect_coverage": 1.0},
    )
    insights = SimpleNamespace(dataframe=aspect_df.copy())
    result = FullAnalysisResult(
        sentiment=sentiment,
        topics=topics,
        aspects=aspects,
        insights=insights,
        timings={
            "sentiment_seconds": 1.0,
            "topic_seconds": 2.0,
            "aspect_seconds": 0.1,
            "insight_seconds": 0.1,
            "total_seconds": 3.2,
        },
    )
    state: dict[str, object] = {}
    persist_full_analysis(state, result, source_signature="abc")

    assert state["analysis_complete"] is True
    assert state["sentiment_complete"] is True
    assert state["topic_complete"] is True
    assert state["aspect_complete"] is True
    assert state["insight_complete"] is True
    assert state["selected_sentiment_model"] == "distilbert"
    assert isinstance(state["results_df"], pd.DataFrame)
