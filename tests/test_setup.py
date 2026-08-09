"""Smoke tests for the Phase 1 project foundation."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_directories_exist() -> None:
    """The repository must contain every top-level Phase 1 directory."""
    required_directories = (
        "pages",
        "src",
        "src/eda",
        "models",
        "data/raw",
        "data/processed",
        "data/sample",
        "tests",
        "docs",
    )

    for relative_path in required_directories:
        assert (PROJECT_ROOT / relative_path).is_dir(), relative_path


def test_source_packages_import() -> None:
    """Every placeholder source package must be importable."""
    import src.aspects  # noqa: F401
    import src.eda  # noqa: F401
    import src.evaluation  # noqa: F401
    import src.ingestion  # noqa: F401
    import src.insights  # noqa: F401
    import src.preprocessing  # noqa: F401
    import src.sentiment  # noqa: F401
    import src.topics  # noqa: F401


def test_sample_dataset_loads() -> None:
    """The bundled development dataset must load and contain review text."""
    sample_path = PROJECT_ROOT / "data" / "sample" / "sample_customer_reviews.csv"
    dataframe = pd.read_csv(sample_path)

    assert not dataframe.empty
    assert "review_text" in dataframe.columns
    assert dataframe["review_text"].notna().all()
