# Technical Design Document
## AI Customer Feedback Intelligence Platform — MVP

**Version:** 1.0  
**Status:** MVP Design  
**Primary Interface:** Streamlit  
**Primary Language:** Python

---

# 1. Purpose

This document defines the technical design for the MVP version of the AI Customer Feedback Intelligence Platform.

The MVP is a portfolio-focused NLP application that allows users to upload a CSV of customer feedback and receive:

- Sentiment predictions
- Topic assignments
- Aspect-level sentiment
- Executive summaries
- Evidence-backed business insights
- Interactive visualizations
- Exportable enriched results

The design intentionally avoids unnecessary production infrastructure and prioritizes a reliable end-to-end implementation.

---

# 2. MVP Scope

## Included

- CSV upload
- Dataset validation
- Text-column selection
- Optional metadata detection
- Basic text preprocessing
- TF-IDF + Logistic Regression baseline
- DistilBERT sentiment inference
- Topic modeling
- Rule-based aspect extraction
- Aspect-level sentiment aggregation
- Executive summary generation
- Streamlit multi-page dashboard
- CSV export
- Model evaluation
- Local execution and simple deployment

## Excluded

- User authentication
- Billing
- Persistent cloud storage
- Team collaboration
- FastAPI microservices
- Kubernetes
- Model registry
- Real-time data streaming
- Scheduled reports
- Multilingual support
- LLM chat assistant
- Enterprise monitoring

---

# 3. System Overview

```text
User
  |
  v
Streamlit Application
  |
  +--> CSV Upload and Validation
  |
  +--> Preprocessing
  |
  +--> Sentiment Analysis
  |
  +--> Topic Modeling
  |
  +--> Aspect-Based Sentiment
  |
  +--> Insight Generation
  |
  +--> Dashboard and Export
```

The MVP runs as a single Python application. Business logic is separated into reusable modules even though the system is deployed as one service.

---

# 4. High-Level Architecture

```text
+--------------------------------------------------+
|                 Streamlit UI                     |
|--------------------------------------------------|
| Upload | Overview | Sentiment | Topics | Aspects |
| Insights | Data Explorer                         |
+-------------------------+------------------------+
                          |
                          v
+--------------------------------------------------+
|              Application Orchestrator            |
+---------+---------------+---------------+--------+
          |               |               |
          v               v               v
+---------------+ +---------------+ +-------------+
| Data Ingestion| | NLP Processing| | Evaluation  |
+---------------+ +---------------+ +-------------+
          |               |
          v               v
+--------------------------------------------------+
|         Enriched Pandas DataFrame                |
+--------------------------+-----------------------+
                           |
                           v
+--------------------------------------------------+
| Dashboard Visualizations and CSV Export          |
+--------------------------------------------------+
```

---

# 5. Recommended Technology Stack

## Core

- Python 3.11+
- Streamlit
- pandas
- NumPy

## Machine Learning

- scikit-learn
- transformers
- PyTorch
- joblib

## NLP

- spaCy or lightweight tokenization utilities
- NLTK only if needed
- BERTopic or NMF, but only one is required for the MVP

## Visualization

- Plotly
- Streamlit native components

## Testing

- pytest

## Optional

- sentence-transformers if BERTopic is selected
- SHAP only if time remains after MVP completion

---

# 6. Repository Structure

```text
customer-feedback-intelligence/
│
├── app.py
├── pages/
│   ├── 1_Overview.py
│   ├── 2_Sentiment.py
│   ├── 3_Topics.py
│   ├── 4_Aspect_Analysis.py
│   ├── 5_Insights.py
│   └── 6_Data_Explorer.py
│
├── src/
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── validator.py
│   │   └── schema.py
│   │
│   ├── preprocessing/
│   │   ├── cleaner.py
│   │   └── text_utils.py
│   │
│   ├── sentiment/
│   │   ├── baseline.py
│   │   ├── transformer.py
│   │   └── inference.py
│   │
│   ├── topics/
│   │   └── topic_model.py
│   │
│   ├── aspects/
│   │   ├── extractor.py
│   │   └── aspect_config.py
│   │
│   ├── insights/
│   │   ├── summaries.py
│   │   └── recommendations.py
│   │
│   ├── evaluation/
│   │   └── metrics.py
│   │
│   └── pipeline.py
│
├── models/
│   ├── logistic_regression.joblib
│   ├── tfidf_vectorizer.joblib
│   └── metadata.json
│
├── data/
│   └── sample/
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_preprocessing.py
│   └── test_pipeline.py
│
├── docs/
│   ├── TECHNICAL_DESIGN.tdd
│   └── EXPERIMENTS.md
│
├── README.md
├── ROADMAP.md
├── requirements.txt
└── .gitignore
```

---

# 7. Core Data Contract

The system converts uploaded data into a canonical DataFrame.

## Required Internal Columns

| Column | Type | Description |
|---|---|---|
| `review_id` | string | Unique identifier |
| `review_text` | string | Customer feedback text |
| `clean_text` | string | Preprocessed text |

## Optional Detected Columns

| Column | Type | Description |
|---|---|---|
| `date` | datetime | Review or ticket date |
| `rating` | numeric | Rating score |
| `product` | string | Product, game, service, or entity |
| `category` | string | Existing category |
| `version` | string | Product or software version |

## Generated Columns

| Column | Type | Description |
|---|---|---|
| `sentiment_label` | string | Positive, neutral, or negative |
| `sentiment_score` | float | Confidence score |
| `topic_id` | integer/string | Assigned topic |
| `topic_label` | string | Human-readable topic |
| `detected_aspects` | list/string | Identified aspects |
| `aspect_sentiment` | object/string | Sentiment by aspect |

---

# 8. Data Ingestion Design

## Responsibilities

- Read uploaded CSV files
- Validate file format
- Detect candidate text columns
- Allow manual override
- Normalize column names
- Remove empty text rows
- Identify duplicates
- Detect optional metadata
- Build the canonical DataFrame

## Text Column Detection

Candidate columns are ranked using:

1. String data type
2. Average text length
3. Percentage of non-empty rows
4. Column-name keywords such as:
   - review
   - text
   - feedback
   - comment
   - content
   - description
   - message

The highest-ranked candidate is suggested to the user, but the user can override it.

## Validation Rules

- File must be CSV.
- At least one text-like column must exist.
- Selected text column must contain usable values.
- Extremely large files may be sampled for interactive performance.
- Encoding errors should produce a clear message.

---

# 9. Preprocessing Design

The preprocessing pipeline should remain simple and model-aware.

## Shared Cleaning

- Convert values to strings
- Trim whitespace
- Normalize repeated spaces
- Remove invalid or empty rows
- Preserve original text

## Classical Model Cleaning

- Lowercase text
- Remove excessive punctuation
- Optional stop-word removal
- Preserve negation terms
- Optional lemmatization

## Transformer Input

- Use minimally cleaned original text
- Preserve punctuation, casing, emojis, and negation
- Apply tokenizer truncation
- Use model-specific padding

The project should not apply aggressive classical preprocessing to DistilBERT input.

---

# 10. Sentiment Analysis Design

## Baseline Model

```text
Clean Text
    |
    v
TF-IDF Vectorizer
    |
    v
Logistic Regression
    |
    v
Sentiment Label
```

### Baseline Outputs

- Predicted class
- Class probability
- Evaluation metrics
- Confusion matrix

## Transformer Model

```text
Original Text
    |
    v
DistilBERT Tokenizer
    |
    v
DistilBERT Classifier
    |
    v
Sentiment Label and Confidence
```

The MVP may use either:

- A fine-tuned DistilBERT model produced during the experiment phase, or
- A suitable pretrained sentiment model during early development

The final portfolio version should clearly state which approach is used.

## Label Contract

The dashboard uses:

- Positive
- Neutral
- Negative

Binary datasets should be handled separately or mapped only when the mapping is defensible.

---

# 11. Topic Modeling Design

The MVP uses one topic-modeling method.

## Preferred First Choice

NMF with TF-IDF is preferred when:

- Implementation speed is important
- Compute is limited
- Transparent keywords are desired

BERTopic may be selected when:

- Topic quality is materially better
- Runtime remains acceptable
- Dependency complexity is manageable

## Topic Outputs

- Topic ID
- Top keywords
- Topic frequency
- Representative reviews
- Average sentiment
- Optional manually edited topic label

## Topic Labeling

Automatic labels may initially use the top two or three keywords.

Example:

```text
Topic 2: battery, charge, power
```

Manual label editing is optional and not required for the MVP.

---

# 12. Aspect-Based Sentiment Design

The MVP uses a rule-based aspect system rather than a trained aspect model.

## Example Configuration

```python
ASPECTS = {
    "price": ["price", "cost", "expensive", "cheap", "value"],
    "quality": ["quality", "durable", "broken", "material"],
    "shipping": ["shipping", "delivery", "package", "arrived"],
    "performance": ["slow", "fast", "lag", "performance"],
    "support": ["support", "service", "agent", "representative"],
}
```

## Processing Flow

```text
Review Text
    |
    v
Keyword or Phrase Matching
    |
    v
Detected Aspects
    |
    v
Reuse Review Sentiment
    |
    v
Aggregate Sentiment by Aspect
```

This approach is intentionally simple, explainable, and realistic for the MVP.

## Limitation

A review may discuss several aspects with different opinions. The MVP may assign the review-level sentiment to all detected aspects. This limitation must be documented.

---

# 13. Insight Generation Design

The insight engine converts aggregated results into concise summaries.

## Inputs

- Sentiment distribution
- Topic frequency
- Average sentiment by topic
- Aspect frequency
- Average sentiment by aspect
- Date-based changes when available
- Representative reviews

## Example Rules

- Most frequent negative topic → top complaint
- Highest-volume positive aspect → top praise
- Largest recent increase in negative mentions → emerging issue
- Aspect with high volume and low sentiment → investigation priority

## Output Requirements

Every insight should include:

- The finding
- The supporting metric
- Example reviews or a filtered view
- Cautious wording

## Example

```text
Shipping is the most common negative aspect, appearing in 24% of
negative reviews. Users frequently mention delayed delivery and
damaged packaging.
```

The MVP does not require free-form generative AI.

---

# 14. Streamlit Application Design

## Page 1 — Upload & Setup

- CSV uploader
- Dataset preview
- Detected text column
- Manual override
- Optional metadata mapping
- Run Analysis button
- Validation results

## Page 2 — Overview

- Review count
- Sentiment distribution
- Topic count
- Top complaint
- Top praised aspect
- Review timeline when available

## Page 3 — Sentiment

- Sentiment distribution
- Confidence distribution
- Sentiment trend
- Filters
- Example reviews

## Page 4 — Topics

- Topic frequency
- Topic keywords
- Average sentiment by topic
- Representative reviews

## Page 5 — Aspect Analysis

- Aspect frequency
- Sentiment by aspect
- Positive and negative aspect rankings
- Supporting reviews

## Page 6 — Insights

- Executive summary
- Top complaints
- Top praises
- Areas for investigation
- Emerging issue section when date data exists

## Page 7 — Data Explorer

- Enriched table
- Search
- Filter
- Sort
- CSV export

---

# 15. Application State

Streamlit session state stores:

- Uploaded DataFrame
- Column mappings
- Cleaned DataFrame
- Selected sentiment model
- Analysis results
- Active filters
- Export-ready DataFrame

Recommended keys:

```python
st.session_state["raw_df"]
st.session_state["clean_df"]
st.session_state["column_mapping"]
st.session_state["analysis_complete"]
st.session_state["results_df"]
st.session_state["topic_summary"]
st.session_state["aspect_summary"]
st.session_state["insights"]
```

The pipeline should avoid rerunning expensive models during every UI interaction.

---

# 16. Pipeline Orchestration

The main pipeline coordinates modules in a fixed sequence.

```python
def run_analysis(df, config):
    canonical_df = prepare_dataset(df, config)
    sentiment_df = run_sentiment(canonical_df, config)
    topic_df, topic_summary = run_topics(sentiment_df, config)
    enriched_df, aspect_summary = run_aspects(topic_df, config)
    insights = generate_insights(
        enriched_df,
        topic_summary,
        aspect_summary,
    )
    return enriched_df, topic_summary, aspect_summary, insights
```

Each module should be independently testable.

---

# 17. Model Artifact Management

The MVP stores model artifacts locally.

## Baseline Artifacts

- TF-IDF vectorizer
- Logistic Regression model
- Label mapping
- Training metadata

## Metadata Example

```json
{
  "model_name": "tfidf_logistic_regression",
  "version": "1.0",
  "training_dataset": "dataset_name",
  "labels": ["negative", "neutral", "positive"],
  "macro_f1": 0.84
}
```

DistilBERT artifacts may be loaded from:

- A local Hugging Face-compatible directory, or
- A published Hugging Face model identifier

A full model registry is not required.

---

# 18. Performance Design

## Main Risks

- Transformer inference can be slow.
- BERTopic can be expensive.
- Large CSVs can overwhelm the UI.
- Repeated Streamlit reruns can duplicate work.

## Mitigations

- Use batching for transformer inference.
- Cache loaded models with `st.cache_resource`.
- Cache deterministic transformations with `st.cache_data`.
- Limit or sample very large uploads.
- Use NMF when BERTopic is too slow.
- Store completed results in session state.
- Display clear progress indicators.

---

# 19. Error Handling

The application should handle:

- Invalid file types
- Empty files
- Missing text columns
- Encoding issues
- Missing optional columns
- Model loading failures
- Topic-modeling failures
- Empty analysis results
- Export failures

Errors should be shown using plain language and should not expose stack traces to normal users.

---

# 20. Testing Strategy

## Unit Tests

Priority areas:

- Column detection
- Dataset validation
- Text cleaning
- Aspect extraction
- Insight rules
- Output schema

## Integration Tests

- Upload to canonical DataFrame
- Canonical DataFrame to sentiment output
- End-to-end pipeline on a small fixture dataset
- Exported output contains generated columns

## Manual Tests

- Dataset with text only
- Dataset with text and rating
- Dataset with text, date, and product
- Missing optional columns
- Invalid upload
- Small and moderately sized datasets

The MVP does not require exhaustive UI automation.

---

# 21. Deployment Design

The preferred deployment target is Streamlit Community Cloud or another simple Python hosting platform.

## Deployment Requirements

- Repository accessible to the deployment platform
- Dependency file
- Streamlit entry point
- Model artifacts available
- Reasonable memory usage
- No local absolute paths
- Clear environment setup

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

A Docker image is optional and should only be added after the application is complete.

---

# 22. Security and Privacy

For the MVP:

- Uploaded data should remain in application memory only.
- Files should not be permanently stored by default.
- The README should warn users not to upload sensitive personal data.
- File-size limits should be enforced.
- User-provided filenames should not be trusted as storage paths.
- Exported results should be created from the in-memory DataFrame.

This project is not designed for confidential production data.

---

# 23. Logging

Use Python's `logging` module for:

- File validation events
- Model loading
- Pipeline start and completion
- Runtime duration
- Recoverable failures

Do not log full customer review text by default.

---

# 24. Configuration

Central configuration should include:

- File-size limit
- Maximum row count
- Sentiment model selection
- Transformer batch size
- Topic count
- Topic-modeling method
- Aspect dictionary
- Confidence threshold

Configuration may be stored in a Python module or YAML file. A complex configuration service is not required.

---

# 25. Key Technical Decisions

## Streamlit Instead of React + API

Reason:
- Faster MVP delivery
- Lower integration overhead
- Strong fit for data science portfolios
- Easier deployment

## Logistic Regression as Baseline

Reason:
- Strong and interpretable NLP baseline
- Fast training and inference
- Useful comparison against DistilBERT

## One Topic Model

Reason:
- Prevents unnecessary experimentation
- Keeps the project focused
- Reduces dependency and runtime complexity

## Rule-Based Aspect Extraction

Reason:
- Explainable
- Fast
- Easy to validate
- Appropriate for a portfolio MVP

## Single-Service Architecture

Reason:
- Easier local development
- Fewer deployment issues
- No benefit from microservices at MVP scale

---

# 26. Known Limitations

- Automatic text-column detection may require manual correction.
- Topic labels may need human interpretation.
- Rule-based aspect extraction will miss synonyms not in the dictionary.
- Review-level sentiment may not accurately represent every aspect.
- DistilBERT performance may degrade across domains.
- Executive summaries are template-driven rather than generative.
- Large datasets may require sampling.

These limitations should be documented in the README and discussed in project interviews.

---

# 27. Definition of Done

The technical implementation is complete when:

- A user can upload a valid CSV.
- The system produces a canonical dataset.
- Logistic Regression and DistilBERT sentiment predictions work.
- Topic modeling produces interpretable themes.
- Aspect-level summaries are generated.
- Evidence-backed insights are displayed.
- All MVP dashboard pages work.
- Results can be exported.
- Critical pipeline tests pass.
- The application runs locally from documented instructions.
- The application is deployed or has a reliable demo.
- No non-MVP infrastructure is required.
