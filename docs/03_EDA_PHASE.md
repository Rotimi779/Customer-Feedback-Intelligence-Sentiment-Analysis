
# 03_EDA_PHASE.md

# Exploratory Data Analysis (EDA) Phase

## Purpose

The Exploratory Data Analysis (EDA) phase provides users with an immediate understanding of their uploaded dataset before any NLP models are executed. This phase validates data quality, highlights trends, and supplies context for the sentiment analysis, topic modeling, and insight generation phases.

The implementation in this document follows the MVP scope defined in **ROADMAP.md** and the architecture in **TECHNICAL_DESIGN.md**.

---

# Objectives

- Summarize the uploaded dataset
- Assess overall data quality
- Detect patterns before model inference
- Support optional metadata (dates, ratings, products)
- Build the Overview page of the Streamlit application

---

# Deliverables

- EDA utility module
- Dataset summary functions
- Interactive visualizations
- Overview dashboard page
- Reusable filtering utilities
- Unit and integration tests

---

# Recommended Implementation Order

1. Create the EDA package
2. Build dataset summary functions
3. Calculate quality metrics
4. Build reusable visualizations
5. Add dataset filters
6. Integrate with Streamlit
7. Write tests
8. Validate with multiple datasets

---

# Files to Create

```text
src/
└── eda/
    ├── __init__.py
    ├── summary.py
    ├── quality.py
    ├── statistics.py
    ├── visualizations.py
    └── filters.py
```

## File Responsibilities

**summary.py**
- Total review count
- Dataset overview
- Summary cards

**quality.py**
- Missing-value analysis
- Duplicate counts
- Empty review detection

**statistics.py**
- Review-length statistics
- Rating statistics
- Time-based summaries

**visualizations.py**
- Plotly charts
- Reusable chart builders

**filters.py**
- Date filters
- Product filters
- Rating filters
- Search utilities

---

# Required Dataset Metrics

Display at minimum:

- Total reviews
- Number of duplicate reviews
- Number of missing reviews
- Average review length
- Median review length
- Dataset size

If optional columns exist, also display:

- Average rating
- Date range
- Number of products
- Number of categories

---

# Visualizations

## Required

- Review length histogram
- Most common words (after light preprocessing)
- Missing-value summary
- Dataset composition cards

## Optional (only when data exists)

- Reviews over time
- Rating distribution
- Reviews by product
- Reviews by category

Visualizations should hide themselves gracefully when the required columns are unavailable.

---

# Filtering

Support interactive filtering by:

- Date
- Product
- Category
- Rating
- Keyword search

Filters should update all charts and metrics consistently.

---

# Streamlit Integration

The **Overview** page should include:

1. KPI summary cards
2. Dataset quality section
3. Interactive charts
4. Filters sidebar
5. Preview of filtered data

This page should complete quickly without requiring any ML inference.

---

# Performance Guidelines

- Cache deterministic calculations using `st.cache_data`
- Avoid recalculating charts after unrelated UI interactions
- Sample only if datasets become too large for smooth interaction

---

# Testing

## Unit Tests

- Summary metrics
- Review-length calculations
- Duplicate detection
- Missing-value statistics
- Optional column handling

## Integration Tests

- Dataset containing only review text
- Dataset with ratings
- Dataset with dates
- Dataset with products
- Dataset with all optional metadata

---

# Suggested Git Commits

- Create EDA package
- Implement summary metrics
- Add quality analysis
- Add Plotly charts
- Implement filters
- Build Overview page
- Add EDA tests

---

# Common Pitfalls

- Assuming optional columns always exist
- Recomputing expensive statistics unnecessarily
- Building charts directly inside Streamlit pages instead of reusable modules
- Mutating the canonical DataFrame during visualization

---

# Definition of Done

This phase is complete when:

- Users immediately understand the uploaded dataset.
- Dataset quality issues are clearly identified.
- Interactive filtering works correctly.
- Visualizations adapt to available metadata.
- The Overview page is fully functional.
- The EDA output is ready for the Sentiment Analysis phase.
