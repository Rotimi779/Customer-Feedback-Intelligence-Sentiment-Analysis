"""Tests for transparent keyword-based topic labels."""

from src.topics.labeling import generate_topic_label, generate_topic_labels


def test_generate_topic_label_is_readable_and_deterministic() -> None:
    label = generate_topic_label(["customer service", "service", "support", "agent"])
    assert label == "Customer Service / Support"


def test_generate_topic_labels_covers_every_topic() -> None:
    labels = generate_topic_labels({0: ["battery", "charge"], 1: ["shipping", "delivery"]})
    assert set(labels) == {0, 1}
    assert labels[0]
    assert labels[1]
