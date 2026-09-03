"""Rule-based aspect-analysis package for the MVP."""

from src.aspects.aggregation import (
    AspectAnalysisResult,
    analyze_aspects,
    build_aspect_summary,
    evaluate_aspect_outputs,
    explode_aspect_mentions,
)
from src.aspects.extraction import AspectExtractor
from src.aspects.sentiment import associate_aspect_sentiment
from src.aspects.utils import (
    DEFAULT_ASPECT_VOCABULARY,
    AspectAnalysisError,
    AspectMatch,
)

__all__ = [
    "AspectAnalysisError",
    "AspectAnalysisResult",
    "AspectExtractor",
    "AspectMatch",
    "DEFAULT_ASPECT_VOCABULARY",
    "analyze_aspects",
    "associate_aspect_sentiment",
    "build_aspect_summary",
    "evaluate_aspect_outputs",
    "explode_aspect_mentions",
]
