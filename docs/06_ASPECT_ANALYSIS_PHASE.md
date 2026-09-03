
# 06_ASPECT_ANALYSIS_PHASE.md

# Aspect Analysis Phase

## Purpose

The Aspect Analysis phase identifies the specific features, products, services, or business areas discussed within customer reviews and determines the sentiment associated with each aspect. Unlike overall sentiment analysis, this phase answers **what** customers are positive or negative about.

This implementation follows the MVP scope defined in **ROADMAP.md**, the architecture described in **TECHNICAL_DESIGN.md**, and the evaluation principles in **EXPERIMENTS.md**.

---

# Objectives

- Identify aspects mentioned in reviews
- Associate sentiment with each detected aspect
- Produce structured aspect-level outputs
- Aggregate aspect sentiment across the dataset
- Prepare results for the Insights phase

---

# Deliverables

- Aspect extraction module
- Aspect sentiment module
- Aggregation utilities
- Aspect visualizations
- Streamlit integration
- Unit and integration tests

---

# Recommended Implementation Order

1. Build aspect extraction pipeline
2. Define aspect vocabulary and rules
3. Link aspects with sentiment predictions
4. Aggregate aspect statistics
5. Build reusable visualizations
6. Integrate into Streamlit
7. Validate with multiple datasets
8. Write tests

---

# Directory Structure

```text
src/
└── aspects/
    ├── __init__.py
    ├── extraction.py
    ├── sentiment.py
    ├── aggregation.py
    ├── visualization.py
    └── utils.py
```

---

# Module Responsibilities

## extraction.py

Responsible for:

- detecting aspect terms
- mapping synonyms to canonical aspect names
- extracting multiple aspects from a review

Example mappings:

- battery, charging → Battery
- support, help desk → Customer Support
- ui, interface → User Interface

---

## sentiment.py

Combine detected aspects with the sentiment predictions from the previous phase.

Each extracted aspect should receive:

- aspect name
- associated sentiment
- confidence (when available)

---

## aggregation.py

Calculate:

- frequency of each aspect
- average sentiment per aspect
- most positive aspects
- most negative aspects
- aspect sentiment distribution

These summaries will power the business insights dashboard.

---

## visualization.py

Create reusable Plotly charts including:

- Top discussed aspects
- Aspect sentiment comparison
- Positive vs negative mentions
- Aspect frequency rankings

---

# Aspect Analysis Workflow

```text
Canonical Reviews
        ↓
Aspect Extraction
        ↓
Sentiment Association
        ↓
Aspect Aggregation
        ↓
Visualization
        ↓
Business Insights
```

---

# Output Schema

Append the following fields to the canonical DataFrame:

- detected_aspects
- aspect_sentiment

Optional fields:

- aspect_confidence

A review may contain multiple aspects.

---

# Aspect Extraction Strategy

For the MVP, implement a rule-based approach using:

- keyword matching
- synonym mapping
- simple phrase detection

Keep the extraction logic modular so more advanced NLP techniques can be added after the MVP.

---

# Streamlit Integration

The Aspect Analysis page should display:

- Most discussed aspects
- Highest-rated aspects
- Lowest-rated aspects
- Aspect sentiment chart
- Filtered review examples

Users should be able to click an aspect and inspect the associated reviews.

---

# Evaluation

Evaluate the system using:

- manual inspection of extracted aspects
- relevance of aspect labels
- correctness of associated sentiment
- usefulness for business interpretation

Record observations and improvements for future iterations.

---

# Testing

## Unit Tests

- aspect extraction
- synonym mapping
- aggregation logic
- sentiment association
- visualization helpers

## Integration Tests

- reviews with one aspect
- reviews with multiple aspects
- reviews with no detectable aspect
- mixed sentiment datasets

Verify that aspect results remain consistent with overall sentiment predictions.

---

# Suggested Git Commits

- Add aspect extraction
- Implement synonym mapping
- Link aspects with sentiment
- Add aggregation utilities
- Build aspect visualizations
- Integrate Aspect Analysis page
- Add aspect analysis tests

---

# Common Pitfalls

- Treating every noun as an aspect
- Ignoring synonyms
- Assigning one sentiment to unrelated aspects
- Overcomplicating the MVP with advanced NLP
- Failing to support multiple aspects in a single review

---

# Definition of Done

This phase is complete when:

- Relevant aspects are extracted from reviews.
- Aspect-level sentiment is generated.
- Aggregated aspect metrics are available.
- Interactive aspect visualizations are functional.
- Outputs are ready for the Business Insights phase.
