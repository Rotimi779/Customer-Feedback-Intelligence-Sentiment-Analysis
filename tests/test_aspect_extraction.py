"""Tests for transparent rule-based aspect extraction."""

from src.aspects.extraction import AspectExtractor


def test_extractor_maps_synonyms_and_multiple_aspects() -> None:
    extractor = AspectExtractor()
    matches = extractor.extract("The help desk was great but the UI is confusing and slow.")
    aspects = [match.aspect for match in matches]

    assert "Customer Support" in aspects
    assert "User Interface" in aspects
    assert "Performance" in aspects


def test_extractor_supports_phrases_without_substring_false_positive() -> None:
    extractor = AspectExtractor({"User Interface": ("ui", "user interface")})

    assert extractor.extract_names("The user interface is clean.") == ["User Interface"]
    assert extractor.extract_names("I built a guide for users.") == []


def test_extractor_returns_empty_for_unmatched_review() -> None:
    assert AspectExtractor().extract_names("Everything arrived yesterday.") == []
