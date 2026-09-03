"""Topic coherence, diversity, coverage, and stability evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.topics.modeling import NMFTopicModel
from src.topics.preprocessing import prepare_topic_corpus
from src.topics.utils import TopicModelError


def _feature_presence(model: NMFTopicModel, texts: list[str]):
    if model.vectorizer is None:
        raise TopicModelError("Topic model is not fitted.")
    matrix = model.vectorizer.transform(texts).tocsr(copy=True)
    matrix.data = np.ones_like(matrix.data)
    return matrix


def topic_coherence_npmi(
    model: NMFTopicModel,
    texts: list[str],
    *,
    top_n: int = 5,
) -> float:
    """Compute mean NPMI over top feature pairs for all topics."""
    if model.vectorizer is None:
        raise TopicModelError("Topic model is not fitted.")
    if len(texts) < 2:
        return 0.0

    binary = _feature_presence(model, texts)
    names = model.vectorizer.get_feature_names_out()
    feature_index = {str(name): index for index, name in enumerate(names)}
    n_docs = float(len(texts))
    scores: list[float] = []

    for keywords in model.topic_keywords.values():
        indices = [feature_index[word] for word in keywords[:top_n] if word in feature_index]
        for left_pos in range(len(indices)):
            for right_pos in range(left_pos + 1, len(indices)):
                left = indices[left_pos]
                right = indices[right_pos]
                left_count = float(binary[:, left].sum())
                right_count = float(binary[:, right].sum())
                both_count = float(binary[:, left].multiply(binary[:, right]).sum())
                if left_count == 0 or right_count == 0:
                    continue
                if both_count == 0:
                    scores.append(-1.0)
                    continue
                p_left = left_count / n_docs
                p_right = right_count / n_docs
                p_both = both_count / n_docs
                pmi = math.log(p_both / (p_left * p_right))
                npmi = pmi / (-math.log(p_both)) if p_both < 1 else 1.0
                scores.append(float(npmi))

    return float(np.mean(scores)) if scores else 0.0


def topic_diversity(model: NMFTopicModel, *, top_n: int = 8) -> float:
    """Measure how many top keywords are unique across topics."""
    words = [
        word
        for keywords in model.topic_keywords.values()
        for word in keywords[:top_n]
    ]
    if not words:
        return 0.0
    return float(len(set(words)) / len(words))


def topic_coverage(model: NMFTopicModel, texts: list[str]) -> float:
    """Measure the share of documents represented by the fitted vocabulary."""
    binary = _feature_presence(model, texts)
    if binary.shape[0] == 0:
        return 0.0
    covered = np.asarray(binary.getnnz(axis=1)).ravel() > 0
    return float(np.mean(covered))


def _topic_set_similarity(
    reference: dict[int, list[str]],
    candidate: dict[int, list[str]],
    *,
    top_n: int,
) -> float:
    """Greedily match topics by Jaccard overlap and average the best matches."""
    reference_sets = {key: set(words[:top_n]) for key, words in reference.items()}
    candidate_sets = {key: set(words[:top_n]) for key, words in candidate.items()}
    remaining = set(candidate_sets)
    matched_scores: list[float] = []

    for ref_set in reference_sets.values():
        best_id: int | None = None
        best_score = -1.0
        for candidate_id in remaining:
            candidate_set = candidate_sets[candidate_id]
            union = ref_set | candidate_set
            score = len(ref_set & candidate_set) / len(union) if union else 1.0
            if score > best_score:
                best_score = score
                best_id = candidate_id
        if best_id is not None:
            remaining.remove(best_id)
            matched_scores.append(best_score)

    return float(np.mean(matched_scores)) if matched_scores else 0.0


def topic_stability(
    model: NMFTopicModel,
    texts: list[str],
    *,
    runs: int = 3,
    top_n: int = 8,
) -> float | None:
    """Refit with alternate seeds and compare top-keyword overlap."""
    if runs <= 1:
        return None

    similarities: list[float] = []
    for offset in range(1, runs):
        candidate = NMFTopicModel(
            replace(model.config, random_state=model.config.random_state + offset)
        )
        candidate.fit(texts)
        similarities.append(
            _topic_set_similarity(
                model.topic_keywords,
                candidate.topic_keywords,
                top_n=top_n,
            )
        )
    return float(np.mean(similarities)) if similarities else None


def evaluate_topic_model(
    model: NMFTopicModel,
    dataframe: pd.DataFrame,
    *,
    stability_runs: int = 3,
) -> dict[str, float | int | bool | None]:
    """Evaluate the fitted topic model using the MVP experiment contract."""
    corpus = prepare_topic_corpus(dataframe)
    weights = model.transform(corpus.texts)
    assignments = weights.argmax(axis=1)

    return {
        "number_of_topics": int(model.config.n_topics),
        "modeled_reviews": int(len(corpus.texts)),
        "average_topic_size": float(len(corpus.texts) / model.config.n_topics),
        "topic_coherence_npmi": topic_coherence_npmi(model, corpus.texts),
        "topic_diversity": topic_diversity(model, top_n=model.config.top_n_words),
        "topic_coverage": topic_coverage(model, corpus.texts),
        "topic_stability": topic_stability(
            model,
            corpus.texts,
            runs=stability_runs,
            top_n=model.config.top_n_words,
        ),
        "all_reviews_assigned": bool(len(assignments) == len(corpus.texts)),
        "manual_interpretability_review_required": True,
    }


def save_topic_evaluation(metrics: dict[str, object], path: str | Path) -> Path:
    """Persist evaluation metrics as readable JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return output
