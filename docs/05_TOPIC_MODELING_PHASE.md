
# 05_TOPIC_MODELING_PHASE.md

# Topic Modeling Phase

## Purpose

The Topic Modeling phase discovers the main themes discussed within customer reviews after sentiment analysis has been completed. Rather than relying on predefined categories, topic modeling groups reviews into meaningful themes that help users understand what customers are talking about at scale.

This implementation follows the MVP defined in **ROADMAP.md**, the architecture in **TECHNICAL_DESIGN.md**, and the evaluation approach in **EXPERIMENTS.md**.

---

# Objectives

- Discover recurring discussion topics
- Assign each review to a topic
- Generate human-readable topic labels
- Measure topic quality
- Visualize topic distributions
- Produce outputs for the Aspect Analysis and Insights phases

---

# Deliverables

- Topic modeling pipeline
- Topic preprocessing module
- Topic labeling utility
- Topic evaluation module
- Interactive visualizations
- Saved topic model
- Unit and integration tests

---

# Recommended Implementation Order

1. Prepare sentiment-enriched dataset
2. Build topic preprocessing
3. Train the topic model
4. Generate keywords for each topic
5. Assign topic IDs to reviews
6. Create readable topic labels
7. Evaluate topic quality
8. Build topic visualizations
9. Integrate into Streamlit
10. Validate using multiple datasets

---

# Directory Structure

```text
src/
└── topics/
    ├── __init__.py
    ├── preprocessing.py
    ├── modeling.py
    ├── labeling.py
    ├── evaluation.py
    ├── visualization.py
    └── utils.py

models/
└── topic_model/
```

---

# Module Responsibilities

## preprocessing.py

Responsible for preparing text for topic modeling by:

- removing empty reviews
- removing duplicate reviews
- tokenization
- stop-word removal
- optional lemmatization

---

## modeling.py

Responsible for:

- training the topic model
- assigning topic IDs
- generating topic probabilities (if available)
- saving the trained model

For the MVP, implement **NMF** as the primary algorithm. BERTopic can be considered later as a post-MVP enhancement.

---

## labeling.py

Generate human-readable labels for each topic using the highest-weighted keywords.

Example:

| Topic ID | Keywords | Label |
|----------|----------|-------|
| 0 | battery, charge, power | Battery Life |
| 1 | support, help, service | Customer Support |

Labels should be concise and understandable.

---

## evaluation.py

Evaluate topics using:

- Topic coherence
- Topic diversity
- Manual interpretability review

Document evaluation results for comparison.

---

## visualization.py

Create reusable Plotly visualizations including:

- Topic frequency bar chart
- Topic distribution pie chart
- Top keywords per topic
- Topic assignment summary

---

# Topic Modeling Workflow

```text
Canonical Reviews
        ↓
Text Preprocessing
        ↓
Topic Model
        ↓
Topic Assignment
        ↓
Topic Labels
        ↓
Evaluation
        ↓
Visualization
```

Each review should receive a `topic_id` and a `topic_label`.

---

# Model Configuration

Recommended starting values:

- Number of topics: experiment with 5–15
- Maximum features: tune based on dataset size
- Minimum document frequency: remove very rare words
- Random state: fixed for reproducibility

The final configuration should balance interpretability and coverage.

---

# Output Schema

Add the following columns to the canonical DataFrame:

- topic_id
- topic_label

If probabilities are available, include:

- topic_confidence

These outputs will be consumed by the Aspect Analysis and Insights phases.

---

# Streamlit Integration

The Topics page should display:

- Number of discovered topics
- Topic frequency chart
- Top keywords for each topic
- Sample reviews per topic
- Topic filters

Users should be able to explore reviews belonging to a selected topic.

---

# Evaluation

Following **EXPERIMENTS.md**, assess:

- Topic coherence
- Topic interpretability
- Coverage across the dataset
- Stability across multiple runs

Record observations and justify the chosen configuration.

---

# Testing

## Unit Tests

- preprocessing
- topic assignment
- label generation
- model persistence
- visualization utilities

## Integration Tests

- small dataset
- large dataset
- mixed sentiment dataset
- unseen dataset

Verify that every review receives a valid topic assignment.

---

# Suggested Git Commits

- Add topic preprocessing
- Implement NMF model
- Generate topic labels
- Add evaluation metrics
- Build topic visualizations
- Integrate Topics page
- Add topic modeling tests

---

# Common Pitfalls

- Selecting too many topics
- Selecting too few topics
- Creating labels that are difficult to interpret
- Training on uncleaned text
- Assuming discovered topics perfectly match business categories

---

# Definition of Done

This phase is complete when:

- Every review is assigned to a topic.
- Human-readable labels are generated.
- Topic quality has been evaluated.
- Interactive topic visualizations are available.
- Topic outputs integrate with the Aspect Analysis phase.
