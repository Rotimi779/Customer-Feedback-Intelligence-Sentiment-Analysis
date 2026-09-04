"""Business-insight generation public API."""

from src.insights.export import (
    build_markdown_report,
    dataframe_to_csv_bytes,
    recommendations_to_csv_bytes,
)
from src.insights.generator import BusinessInsightsResult, generate_business_insights
from src.insights.trends import TrendAnalysisResult, analyze_trends
from src.insights.utils import InsightFinding, InsightRecommendation, InsightsError

__all__ = [
    "BusinessInsightsResult",
    "InsightFinding",
    "InsightRecommendation",
    "InsightsError",
    "TrendAnalysisResult",
    "analyze_trends",
    "build_markdown_report",
    "dataframe_to_csv_bytes",
    "generate_business_insights",
    "recommendations_to_csv_bytes",
]
