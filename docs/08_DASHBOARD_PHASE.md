
# 08_DASHBOARD_PHASE.md

# Dashboard Phase

## Purpose

The Dashboard phase combines the outputs from Data Ingestion, Exploratory Data Analysis, Sentiment Analysis, Topic Modeling, Aspect Analysis, and Business Insights into one cohesive Streamlit application.

The goal of this phase is not to redesign or expand the analytical pipeline. Instead, it focuses on presenting the existing outputs clearly, organizing the user journey, maintaining application state, and allowing users to explore results through a polished multi-page interface.

This implementation follows the MVP scope defined in **ROADMAP.md** and the application architecture described in **TECHNICAL_DESIGN.md**.

---

# Objectives

- Build the complete multi-page Streamlit dashboard
- Connect all completed analysis modules
- Create a clear end-to-end user workflow
- Maintain uploaded data and model outputs across pages
- Add global filters and consistent visualizations
- Support result previews and exports
- Handle loading, empty, and error states
- Prepare the application for final testing and deployment

---

# Deliverables

- Functional Streamlit application shell
- Multi-page navigation
- Upload and dataset configuration page
- Overview and EDA page
- Sentiment Analysis page
- Topic Modeling page
- Aspect Analysis page
- Business Insights page
- Export section
- Shared UI components
- Session-state management
- Dashboard integration tests

---

# Dependencies

This phase depends on the following implementation phases being substantially complete:

- `02_DATA_INGESTION_PHASE.md`
- `03_EDA_PHASE.md`
- `04_SENTIMENT_ANALYSIS_PHASE.md`
- `05_TOPIC_MODELING_PHASE.md`
- `06_ASPECT_ANALYSIS_PHASE.md`
- `07_INSIGHTS_PHASE.md`

The dashboard should consume outputs from these modules rather than duplicate their data-processing logic.

---

# Recommended Implementation Order

1. Review the completed module interfaces
2. Create the Streamlit application shell
3. Configure page navigation
4. Define session-state keys
5. Build reusable UI components
6. Build the Upload page
7. Build the Overview page
8. Build the Sentiment page
9. Build the Topics page
10. Build the Aspects page
11. Build the Insights page
12. Add global filtering
13. Add export controls
14. Add loading and error states
15. Test the complete dashboard workflow

---

# Recommended Directory Structure

```text
customer-feedback-intelligence/
│
├── app.py
│
├── pages/
│   ├── 1_Upload_Data.py
│   ├── 2_Overview.py
│   ├── 3_Sentiment_Analysis.py
│   ├── 4_Topic_Modeling.py
│   ├── 5_Aspect_Analysis.py
│   └── 6_Business_Insights.py
│
├── src/
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── components.py
│   │   ├── navigation.py
│   │   ├── state.py
│   │   ├── filters.py
│   │   ├── formatting.py
│   │   └── errors.py
│   │
│   ├── ingestion/
│   ├── eda/
│   ├── sentiment/
│   ├── topics/
│   ├── aspects/
│   └── insights/
│
├── tests/
│   ├── test_dashboard_state.py
│   ├── test_dashboard_filters.py
│   └── test_dashboard_workflow.py
│
└── .streamlit/
    └── config.toml
```

The exact filenames can vary, but the separation between presentation logic and analysis logic should remain clear.

---

# Application Workflow

The MVP dashboard should guide the user through a simple sequence:

```text
Upload CSV
    ↓
Confirm Review Column
    ↓
Validate and Prepare Dataset
    ↓
Review Dataset Overview
    ↓
Run Analysis
    ↓
Explore Sentiment
    ↓
Explore Topics
    ↓
Explore Aspects
    ↓
Read Business Insights
    ↓
Export Results
```

The user should always understand:

- what has already been completed
- what step comes next
- whether analysis is currently running
- whether the results shown are based on filtered or complete data

---

# Application Shell

## app.py Responsibilities

The root application should:

- configure the Streamlit page
- define the application title and icon
- initialize session state
- display the project introduction
- explain the analysis workflow
- provide navigation guidance
- verify whether a dataset has been uploaded
- redirect or warn users when required data is missing

Example responsibilities:

```python
import streamlit as st

from src.dashboard.state import initialize_session_state

st.set_page_config(
    page_title="AI Customer Feedback Intelligence Platform",
    page_icon="💬",
    layout="wide",
)

initialize_session_state()

st.title("AI Customer Feedback Intelligence Platform")
st.write(
    "Upload customer reviews, analyze sentiment, discover topics, "
    "evaluate product aspects, and generate business insights."
)
```

The root file should remain lightweight. It should not contain the full analytical pipeline.

---

# Page Navigation

The dashboard should use Streamlit's multi-page application structure.

Recommended pages:

1. Upload Data
2. Overview
3. Sentiment Analysis
4. Topic Modeling
5. Aspect Analysis
6. Business Insights

## Navigation Rules

- Upload Data must always be accessible.
- Analysis pages should require a validated canonical DataFrame.
- Topic and Aspect pages should require completed model outputs.
- Business Insights should require completed sentiment, topic, and aspect results.
- Pages should display a clear warning when prerequisites are missing.

Example:

```python
if st.session_state.get("canonical_df") is None:
    st.warning("Upload and validate a dataset before opening this page.")
    st.stop()
```

Avoid silently failing or showing empty charts when required inputs are unavailable.

---

# Session-State Design

Streamlit reruns the active page whenever a user interacts with a widget. Session state is therefore required to preserve the uploaded dataset, filters, model outputs, and analysis status.

## Recommended Session-State Keys

```text
uploaded_file_name
raw_df
canonical_df
filtered_df
column_mapping
validation_report
analysis_complete
analysis_running
sentiment_results
topic_results
aspect_results
insight_results
active_filters
selected_model
selected_topic
selected_aspect
last_error
```

## State Initialization

Create a dedicated initialization function in `src/dashboard/state.py`.

```python
import streamlit as st

DEFAULT_STATE = {
    "uploaded_file_name": None,
    "raw_df": None,
    "canonical_df": None,
    "filtered_df": None,
    "column_mapping": {},
    "validation_report": None,
    "analysis_complete": False,
    "analysis_running": False,
    "sentiment_results": None,
    "topic_results": None,
    "aspect_results": None,
    "insight_results": None,
    "active_filters": {},
    "selected_model": None,
    "selected_topic": None,
    "selected_aspect": None,
    "last_error": None,
}

def initialize_session_state() -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value
```

## Reset Behaviour

Provide a reset action when the user uploads a new dataset.

Reset:

- filtered data
- sentiment outputs
- topic outputs
- aspect outputs
- insight outputs
- active selections
- analysis status

Do not keep stale model outputs from a previous file.

---

# Shared Dashboard Components

Create reusable functions in `src/dashboard/components.py`.

Recommended components:

- KPI cards
- section headers
- empty-state messages
- status banners
- analysis progress display
- filtered data preview
- download controls
- result tables
- model information box
- warning and error panels

## KPI Cards

Use `st.metric` for compact summaries such as:

- Total Reviews
- Positive Sentiment
- Negative Sentiment
- Topics Found
- Aspects Found
- Average Confidence

Keep metric formatting consistent across pages.

## Status Banners

Use clear status messages:

- Dataset uploaded
- Validation passed
- Analysis running
- Analysis complete
- Analysis failed
- No matching records after filtering

---

# Upload Data Page

## Purpose

The Upload Data page handles file selection, schema detection, validation, and creation of the canonical DataFrame.

## Required Components

- CSV file uploader
- file details
- raw data preview
- automatically detected text column
- manual text-column selector
- optional metadata selectors
- validation results
- continue or run-analysis button

## Recommended Layout

```text
Page Title
    ↓
Upload Control
    ↓
File Summary
    ↓
Column Detection
    ↓
Data Preview
    ↓
Validation Messages
    ↓
Confirm Dataset
```

## Upload Interaction

```python
uploaded_file = st.file_uploader(
    "Upload customer feedback",
    type=["csv"],
)
```

After upload:

1. Load the file using the ingestion module.
2. Display a small preview.
3. Suggest the review-text column.
4. Allow the user to override the suggestion.
5. Detect optional metadata columns.
6. Validate the final selection.
7. Save the canonical DataFrame in session state.

## File Summary

Display:

- filename
- number of rows
- number of columns
- detected review column
- optional metadata found
- duplicate count
- missing review count

## Confirm Dataset Button

Do not start analysis automatically when the file is uploaded.

Provide a button such as:

```text
Confirm Dataset
```

Once confirmed:

- create the canonical DataFrame
- save it in session state
- reset stale outputs
- allow access to the Overview page

---

# Overview Page

## Purpose

The Overview page presents the dataset's descriptive statistics and quality checks before model results.

## Required Sections

### Dataset Summary

Display:

- total reviews
- average review length
- median review length
- duplicate count
- missing review count

When available:

- date range
- average rating
- number of products
- number of categories

### Data Quality

Show:

- missing-value summary
- duplicate summary
- empty-review count
- optional metadata coverage

### Exploratory Visualizations

Include:

- review length distribution
- most common words
- rating distribution, when available
- review volume over time, when available
- reviews by product or category, when available

### Data Preview

Display a filtered sample of the canonical DataFrame.

Avoid showing thousands of rows by default.

---

# Sentiment Analysis Page

## Purpose

The Sentiment Analysis page displays the results produced by the selected production sentiment model.

## Required Sections

### Model Summary

Display:

- model name
- model type
- label set
- average confidence
- number of reviews analyzed

### KPI Metrics

Display:

- positive review percentage
- neutral review percentage
- negative review percentage
- average sentiment confidence

### Visualizations

Include:

- sentiment distribution bar chart
- sentiment proportion chart
- sentiment over time, when dates exist
- sentiment by product, when product metadata exists
- confidence distribution

### Review Explorer

Allow users to:

- filter by sentiment label
- filter by confidence
- search review text
- inspect individual predictions

Recommended columns:

- review_text
- sentiment_label
- sentiment_score
- optional product
- optional date

## Model Comparison

The MVP may include a compact comparison between Logistic Regression and DistilBERT using the evaluation results from `EXPERIMENTS.md`.

Display:

- accuracy
- macro F1
- weighted F1
- inference time
- selected production model

Do not retrain models from the dashboard.

---

# Topic Modeling Page

## Purpose

The Topic Modeling page allows users to explore the main themes discovered in the dataset.

## Required Sections

### Topic Summary

Display:

- number of topics
- largest topic
- smallest topic
- percentage of reviews assigned

### Topic Frequency

Include a bar chart showing review count by topic.

### Topic Keywords

Display a table with:

- topic ID
- topic label
- top keywords
- review count
- percentage of dataset

### Topic Explorer

Allow the user to select a topic and view:

- topic label
- top keywords
- topic size
- sentiment distribution within the topic
- example reviews

### Topic Filtering

Allow filters for:

- selected topic
- sentiment label
- product
- date range

## Empty Topic Handling

If a topic contains too few reviews or the model cannot create meaningful topics, display a clear explanation instead of an empty visualization.

---

# Aspect Analysis Page

## Purpose

The Aspect Analysis page explains which product or service attributes customers mention and how they feel about each one.

## Required Sections

### Aspect Summary

Display:

- number of unique aspects
- most discussed aspect
- most positive aspect
- most negative aspect

### Aspect Frequency

Display a bar chart of aspect mention counts.

### Aspect Sentiment Comparison

Display positive, neutral, and negative mentions for each aspect.

A grouped or stacked bar chart is suitable.

### Aspect Ranking

Create tables for:

- top positive aspects
- top negative aspects
- most frequently discussed aspects

### Aspect Explorer

Allow users to select an aspect and inspect:

- mention count
- sentiment distribution
- average confidence
- example reviews
- associated topics

## Multi-Aspect Reviews

A review may appear under more than one aspect. Make this clear in the interface and avoid treating aspect counts as unique-review counts unless explicitly calculated that way.

---

# Business Insights Page

## Purpose

The Business Insights page presents the most important findings in plain language.

## Required Sections

### Executive Summary

Present a concise summary covering:

- total reviews analyzed
- overall sentiment
- dominant topics
- strongest positive aspect
- largest pain point
- highest-priority improvement area

### Key Findings

Display a small set of evidence-backed findings.

Example structure:

```text
Finding
Evidence
Business Interpretation
```

### Recommendations

Display rule-based recommendations generated during the Insights phase.

Each recommendation should include:

- recommendation title
- supporting metric
- affected topic or aspect
- priority level
- short explanation

### Trends

When date metadata exists, include:

- sentiment trend
- rising topic
- worsening aspect
- review-volume trend

If date metadata does not exist, hide the section and explain that trend analysis requires a date column.

### Export Controls

Allow the user to download:

- enriched review dataset
- summary report
- recommendation table

---

# Global Filtering

Filters should remain consistent across analysis pages.

Recommended filters:

- date range
- product
- category
- rating
- sentiment label
- topic
- aspect
- text search

## Filtering Rules

- Apply filters to a copy of the canonical or enriched DataFrame.
- Do not mutate the original DataFrame.
- Show the number of matching reviews.
- Provide a clear reset button.
- Indicate when charts use filtered data.
- Handle zero-result filters gracefully.

## Filter Utility

Place reusable filtering logic in `src/dashboard/filters.py`.

Example interface:

```python
def apply_filters(
    dataframe,
    date_range=None,
    products=None,
    categories=None,
    ratings=None,
    sentiments=None,
    topics=None,
    aspects=None,
    search_text=None,
):
    ...
```

The function should return a filtered copy of the DataFrame.

---

# Running the Analysis Pipeline

The dashboard should provide one clear action for running the complete analysis pipeline.

Recommended button:

```text
Run Full Analysis
```

## Pipeline Sequence

1. Validate canonical DataFrame
2. Run sentiment inference
3. Run topic modeling or load the configured model
4. Run aspect extraction
5. Aggregate results
6. Generate insights
7. Save all outputs to session state

## Progress Display

Use a progress bar and stage-level status messages.

Example:

```text
Preparing reviews
Running sentiment analysis
Discovering topics
Extracting aspects
Generating insights
Analysis complete
```

Do not display misleading progress percentages if stage durations are unpredictable. Stage-based updates are sufficient for the MVP.

## Failure Handling

If a stage fails:

- capture the exception
- log the error
- show a user-friendly message
- preserve valid earlier results where practical
- avoid displaying incomplete outputs as final results

---

# Caching Strategy

Use Streamlit caching carefully.

## st.cache_data

Suitable for:

- file loading
- deterministic summaries
- filtered datasets
- chart-ready aggregations
- exported CSV conversion

## st.cache_resource

Suitable for:

- sentiment model loading
- tokenizer loading
- topic model loading
- NLP model initialization

Example:

```python
@st.cache_resource
def load_sentiment_model():
    ...
```

Do not cache mutable session-state objects directly.

---

# Visualization Standards

Use Plotly consistently for interactive charts.

## Chart Requirements

Every chart should include:

- descriptive title
- labelled axes
- readable legend
- hover information
- sensible sorting
- empty-data handling

## Recommended Chart Types

| Analysis | Recommended Chart |
|---|---|
| Sentiment distribution | Bar chart |
| Sentiment proportions | Donut or bar chart |
| Review volume over time | Line chart |
| Topic frequency | Horizontal bar chart |
| Topic keywords | Table |
| Aspect sentiment | Stacked bar chart |
| Rating distribution | Histogram or bar chart |
| Confidence distribution | Histogram |

Avoid adding decorative charts that do not answer a clear question.

---

# Formatting and Labels

Create formatting utilities in `src/dashboard/formatting.py`.

Recommended functions:

- percentage formatting
- confidence formatting
- date formatting
- large-number formatting
- label capitalization
- safe text truncation

Example:

```python
def format_percentage(value: float) -> str:
    return f"{value:.1%}"
```

Use consistent terms throughout the dashboard:

- Positive
- Neutral
- Negative
- Topic
- Aspect
- Confidence
- Reviews Analyzed

---

# Empty, Loading, and Error States

A polished MVP must clearly handle states where results are not yet available.

## Empty States

Examples:

- No dataset uploaded
- Dataset validated but analysis not run
- No date column available
- No topic matches selected filters
- No aspect detected
- No reviews remain after filtering

## Loading States

Use:

- `st.spinner`
- progress bars
- stage messages
- disabled buttons where appropriate

## Error States

Messages should explain:

- what failed
- whether any results remain available
- what the user can do next

Avoid displaying raw stack traces in the user interface.

---

# Download and Export Controls

The dashboard should allow users to export completed results.

## Enriched Dataset

Recommended fields:

- review_id
- review_text
- clean_text
- sentiment_label
- sentiment_score
- topic_id
- topic_label
- detected_aspects
- aspect_sentiment
- optional metadata columns

## Summary Export

The MVP can export a Markdown, text, or CSV summary containing:

- dataset overview
- sentiment breakdown
- main topics
- top aspects
- recommendations

## CSV Conversion

```python
@st.cache_data
def dataframe_to_csv(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")
```

Use descriptive filenames that include the analysis type.

---

# Responsive Layout

Use Streamlit columns carefully.

Recommended patterns:

- three or four KPI cards in one row
- chart and explanation in two columns
- full-width tables below charts
- filters in the sidebar

Avoid placing wide tables inside narrow columns.

The dashboard should remain usable on a standard laptop screen without excessive horizontal scrolling.

---

# Accessibility and Usability

For the MVP:

- use clear headings
- avoid relying only on colour to communicate meaning
- provide chart labels
- keep instructions near relevant controls
- use plain language
- maintain consistent terminology
- avoid overcrowded pages

Important findings should be communicated in text as well as charts.

---

# Performance Guidelines

- Load models once using `st.cache_resource`
- Cache deterministic transformations
- Avoid recomputing the entire pipeline after filter changes
- Separate model execution from visualization updates
- Limit table previews
- Use batched sentiment inference
- Avoid rendering too many charts at once
- Sample only when necessary for responsiveness

Filtering should operate on saved results, not rerun the ML pipeline.

---

# Testing Strategy

## Unit Tests

Test:

- session-state initialization
- state reset
- filter functions
- formatting functions
- CSV export conversion
- prerequisite checks
- empty-state helpers

## Integration Tests

Test:

- upload to canonical DataFrame
- canonical DataFrame to complete analysis
- complete analysis to dashboard display
- filter updates across pages
- new upload resetting previous results
- exports containing expected columns

## Manual Dashboard Tests

Verify:

- every page loads
- navigation works
- warnings appear when prerequisites are missing
- charts update after filtering
- reset controls work
- progress indicators appear
- empty results do not crash the app
- downloads contain correct data
- optional metadata sections hide correctly
- analysis results persist across page changes

---

# Suggested Test Scenarios

## Scenario 1: Text-Only Dataset

Expected behaviour:

- ingestion succeeds
- overview displays text metrics
- sentiment, topics, and aspects run
- date and rating charts remain hidden
- insights still generate

## Scenario 2: Full Metadata Dataset

Expected behaviour:

- all filters are available
- date trends display
- rating summaries display
- product and category comparisons display

## Scenario 3: Invalid Upload

Expected behaviour:

- validation error appears
- analysis button remains unavailable
- no analysis state is created

## Scenario 4: No Filter Matches

Expected behaviour:

- zero-result message appears
- charts do not crash
- reset-filter control is visible

## Scenario 5: New Dataset Upload

Expected behaviour:

- previous results are cleared
- filters reset
- new dataset requires confirmation
- no stale charts remain

---

# Suggested Git Commits

Recommended milestones:

- Create Streamlit application shell
- Add dashboard session-state management
- Build shared UI components
- Implement Upload Data page
- Implement Overview page
- Implement Sentiment Analysis page
- Implement Topic Modeling page
- Implement Aspect Analysis page
- Implement Business Insights page
- Add global filters
- Add export controls
- Add dashboard loading and error states
- Add dashboard integration tests

Each commit should represent a working improvement rather than combining the complete dashboard into one large commit.

---

# Common Pitfalls

- Placing analytical logic directly inside Streamlit pages
- Rerunning models after every filter interaction
- Failing to reset outputs after a new upload
- Assuming optional metadata always exists
- Using inconsistent labels across pages
- Showing raw errors to users
- Displaying empty charts without explanation
- Mutating the canonical DataFrame
- Allowing users to access results pages before analysis completes
- Overloading the dashboard with unnecessary visualizations
- Treating aspect mentions as unique review counts
- Using filters that behave differently across pages

---

# Phase Completion Checklist

## Application Structure

- [ ] Root Streamlit application is configured
- [ ] Multi-page navigation works
- [ ] Dashboard helper modules exist
- [ ] Session state initializes correctly

## Upload and State

- [ ] CSV upload is connected to the ingestion pipeline
- [ ] Column selection works
- [ ] Validation messages display
- [ ] New uploads reset stale results
- [ ] Canonical DataFrame persists across pages

## Analysis Pages

- [ ] Overview page is complete
- [ ] Sentiment page is complete
- [ ] Topic Modeling page is complete
- [ ] Aspect Analysis page is complete
- [ ] Business Insights page is complete

## Interaction

- [ ] Global filters work
- [ ] Reset filters works
- [ ] Review explorers work
- [ ] Empty states display correctly
- [ ] Loading and progress states display correctly
- [ ] Errors are handled clearly

## Export

- [ ] Enriched dataset can be downloaded
- [ ] Summary results can be downloaded
- [ ] Exported fields match the canonical output schema

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual workflow tests pass
- [ ] Optional metadata is handled correctly
- [ ] No page crashes when results are missing

---

# Definition of Done

The Dashboard phase is complete when:

- A user can upload and validate a CSV file.
- The canonical dataset persists across the application.
- The complete analysis pipeline can be launched from the interface.
- Sentiment, topic, aspect, and insight outputs are displayed on dedicated pages.
- Global filters update saved results without rerunning models.
- Optional metadata is handled gracefully.
- Loading, empty, and error states are clearly communicated.
- Users can inspect reviews and download final outputs.
- The dashboard works from upload through export without manual code execution.
- The application is ready for the Testing and Deployment phase.
