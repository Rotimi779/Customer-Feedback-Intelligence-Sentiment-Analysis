"""Tests for model-aware sentiment preprocessing."""

from src.sentiment.preprocessing import (
    prepare_classical_text,
    prepare_transformer_text,
)


def test_classical_preprocessing_lowercases_and_collapses_punctuation() -> None:
    assert prepare_classical_text("  NOT   good!!!!  ") == "not good!"


def test_transformer_preprocessing_preserves_case_and_punctuation() -> None:
    assert prepare_transformer_text("  NOT   good!!!!  ") == "NOT good!!!!"
