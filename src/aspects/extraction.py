"""Rule-based aspect extraction using keywords, synonyms, and phrases."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from src.aspects.utils import (
    DEFAULT_ASPECT_VOCABULARY,
    AspectMatch,
    normalize_aspect_text,
    validate_aspect_vocabulary,
)


class AspectExtractor:
    """Extract canonical aspects with transparent vocabulary rules."""

    def __init__(
        self,
        vocabulary: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.vocabulary = validate_aspect_vocabulary(
            vocabulary or DEFAULT_ASPECT_VOCABULARY
        )
        self._patterns = {}
        for aspect, terms in self.vocabulary.items():
            compiled_patterns = []
            for term in sorted(terms, key=len, reverse=True):
                escaped_term = re.escape(term).replace(r"\ ", r"\s+")
                pattern = re.compile(rf"(?<!\w){escaped_term}(?!\w)")
                compiled_patterns.append((term, pattern))
            self._patterns[aspect] = tuple(compiled_patterns)

    def extract(self, text: object) -> list[AspectMatch]:
        """Return every canonical aspect detected in one review."""
        normalized = normalize_aspect_text(text)
        if not normalized:
            return []

        matches: list[AspectMatch] = []
        for aspect, patterns in self._patterns.items():
            matched_terms = tuple(
                term for term, pattern in patterns if pattern.search(normalized)
            )
            if matched_terms:
                matches.append(AspectMatch(aspect=aspect, matched_terms=matched_terms))
        return matches

    def extract_names(self, text: object) -> list[str]:
        """Return canonical aspect names in vocabulary order."""
        return [match.aspect for match in self.extract(text)]
