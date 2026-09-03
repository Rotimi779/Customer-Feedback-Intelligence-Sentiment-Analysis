
# 04_SENTIMENT_ANALYSIS_PHASE.md

# Sentiment Analysis Phase

## Purpose

This phase implements the first machine learning capability of the AI Customer Feedback Intelligence Platform. The goal is to classify each customer review as **Positive**, **Neutral**, or **Negative**, compare a traditional machine learning baseline with a transformer model, and produce standardized outputs for downstream topic modeling, aspect analysis, and business insights.

This implementation follows the MVP defined in **ROADMAP.md**, the architecture in **TECHNICAL_DESIGN.md**, and the evaluation strategy in **EXPERIMENTS.md**.

---

# Objectives

- Build a complete sentiment analysis pipeline
- Implement a TF-IDF + Logistic Regression baseline
- Implement a DistilBERT model
- Compare both approaches using common evaluation metrics
- Select the production model
- Generate standardized sentiment outputs
- Integrate inference into the Streamlit application

---

# Deliverables

- Sentiment preprocessing module
- Baseline model
- DistilBERT model
- Training scripts
- Inference pipeline
- Evaluation reports
- Saved production model
- Streamlit integration
- Unit and integration tests

---

# Recommended Implementation Order

1. Prepare labelled datasets
2. Build preprocessing pipeline
3. Train Logistic Regression baseline
4. Evaluate baseline
5. Fine-tune DistilBERT
6. Evaluate DistilBERT
7. Compare both models
8. Select production model
9. Build inference pipeline
10. Integrate with Streamlit
11. Test the complete workflow

---

# Directory Structure

```text
src/
└── sentiment/
    ├── __init__.py
    ├── preprocessing.py
    ├── train_baseline.py
    ├── train_transformer.py
    ├── inference.py
    ├── evaluation.py
    ├── metrics.py
    └── utils.py

models/
├── logistic_regression/
└── distilbert/
```

---

# Module Responsibilities

## preprocessing.py

Responsible for:

- cleaning review text
- removing invalid records
- optional lemmatization for classical ML
- preparing transformer inputs

The preprocessing used during inference must match training exactly.

---

## train_baseline.py

Train a Logistic Regression classifier using:

- TF-IDF Vectorizer
- Logistic Regression

Save:

- vectorizer
- trained model
- training metadata

---

## train_transformer.py

Fine-tune DistilBERT on the selected sentiment dataset.

Responsibilities:

- tokenizer
- dataset preparation
- model training
- checkpoint saving
- final model export

---

## inference.py

Provides a single prediction interface.

Input:

- review text

Output:

- sentiment label
- confidence score

The rest of the application should interact only with this interface.

---

## evaluation.py

Generate:

- confusion matrix
- classification report
- accuracy
- precision
- recall
- F1 score

Store results for comparison.

---

# Dataset Preparation

The labelled dataset should:

- contain review text
- contain sentiment labels
- remove duplicates
- remove empty reviews
- balance classes where practical

Recommended split:

- Training: 70%
- Validation: 15%
- Testing: 15%

---

# Baseline Model

## Algorithm

- TF-IDF Vectorizer
- Logistic Regression

Purpose:

- establish an interpretable baseline
- provide fast inference
- measure transformer improvement

Tune parameters such as:

- ngram range
- max_features
- regularization strength

---

# DistilBERT Model

Use DistilBERT for sequence classification.

Training considerations:

- tokenizer
- batch size
- learning rate
- epochs
- early stopping

Save:

- tokenizer
- configuration
- trained weights

---

# Model Comparison

Compare models using:

| Metric | Logistic Regression | DistilBERT |
|---------|--------------------|------------|
| Accuracy | ✓ | ✓ |
| Precision | ✓ | ✓ |
| Recall | ✓ | ✓ |
| Macro F1 | ✓ | ✓ |
| Weighted F1 | ✓ | ✓ |
| Training Time | ✓ | ✓ |
| Inference Time | ✓ | ✓ |

The production model should be chosen based on overall performance and practical inference speed, consistent with the MVP goals.

---

# Inference Pipeline

```text
Review
   ↓
Preprocessing
   ↓
Selected Model
   ↓
Prediction
   ↓
Confidence Score
   ↓
Store Results
```

Generated columns:

- sentiment_label
- sentiment_score

These columns become inputs for later phases.

---

# Streamlit Integration

The sentiment stage should begin only after successful ingestion and EDA.

Display:

- model in use
- prediction progress
- sentiment distribution
- average confidence
- downloadable results

If inference fails, display a user-friendly error rather than stopping the application.

---

# Evaluation Metrics

Following **EXPERIMENTS.md**, report:

- Accuracy
- Precision
- Recall
- Macro F1
- Weighted F1
- Confusion Matrix

Document runtime observations:

- training duration
- inference speed
- model size

---

# Testing

## Unit Tests

- preprocessing
- tokenizer loading
- TF-IDF generation
- prediction output
- confidence scores

## Integration Tests

- baseline inference
- DistilBERT inference
- model loading
- Streamlit workflow
- prediction on unseen reviews

---

# Suggested Git Commits

- Add preprocessing pipeline
- Train Logistic Regression baseline
- Add evaluation metrics
- Fine-tune DistilBERT
- Compare models
- Build inference pipeline
- Integrate Streamlit predictions
- Add sentiment tests

---

# Common Pitfalls

- Using different preprocessing during training and inference
- Data leakage between train and test sets
- Comparing models with different datasets
- Ignoring class imbalance
- Returning labels without confidence scores

---

# Definition of Done

This phase is complete when:

- Both sentiment models have been trained.
- Evaluation metrics are generated.
- A production model has been selected.
- The application predicts sentiment for uploaded reviews.
- Confidence scores are produced.
- Results integrate seamlessly with downstream topic modeling and aspect analysis.
