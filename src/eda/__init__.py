"""Public API for exploratory data analysis."""

from src.eda.filters import DatasetFilters, FilterOptions, apply_filters, get_filter_options
from src.eda.quality import (
    DataQualitySummary,
    count_duplicate_reviews,
    count_empty_reviews,
    missing_value_summary,
    summarize_quality,
)
from src.eda.statistics import (
    DEFAULT_STOP_WORDS,
    RatingStatistics,
    ReviewLengthStatistics,
    add_review_length_columns,
    calculate_common_words,
    calculate_rating_statistics,
    calculate_review_length_statistics,
    infer_time_frequency,
    summarize_reviews_over_time,
)
from src.eda.summary import DatasetSummary, build_dataset_summary, format_bytes
from src.eda.visualizations import (
    build_common_words_chart,
    build_group_distribution_chart,
    build_missing_values_chart,
    build_rating_distribution_chart,
    build_review_length_histogram,
    build_reviews_over_time_chart,
)

__all__ = [
    "DEFAULT_STOP_WORDS",
    "DataQualitySummary",
    "DatasetFilters",
    "DatasetSummary",
    "FilterOptions",
    "RatingStatistics",
    "ReviewLengthStatistics",
    "add_review_length_columns",
    "apply_filters",
    "build_common_words_chart",
    "build_dataset_summary",
    "build_group_distribution_chart",
    "build_missing_values_chart",
    "build_rating_distribution_chart",
    "build_review_length_histogram",
    "build_reviews_over_time_chart",
    "calculate_common_words",
    "calculate_rating_statistics",
    "calculate_review_length_statistics",
    "count_duplicate_reviews",
    "count_empty_reviews",
    "format_bytes",
    "get_filter_options",
    "infer_time_frequency",
    "missing_value_summary",
    "summarize_quality",
    "summarize_reviews_over_time",
]
