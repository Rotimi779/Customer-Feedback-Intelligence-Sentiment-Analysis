# EXPERIMENTS
## AI Customer Feedback Intelligence Platform — MVP

**Purpose:** Define the experiments required to evaluate the NLP components included in the MVP. This document focuses only on experiments that will actually be implemented.

---

# 1. Research Goal

Evaluate whether a transformer-based sentiment model (DistilBERT) provides meaningful improvements over a strong classical NLP baseline (TF-IDF + Logistic Regression) while producing useful business insights from customer feedback.

---

# 2. Research Questions

## Primary
- Does DistilBERT outperform a TF-IDF + Logistic Regression baseline on sentiment classification?

## Secondary
- Which model offers the best balance of accuracy and inference speed?
- How well do discovered topics align with human interpretation?
- Do aspect summaries provide actionable insights?
- How well does the pipeline generalize across different customer-review domains?

---

# 3. Datasets

Use at least two public datasets.

## Primary Dataset
- Amazon Product Reviews (or equivalent)

## Secondary Dataset
Choose one:
- Steam Reviews
- Twitter Airline Sentiment
- Customer Support Tickets

Purpose:
- Validate the pipeline on multiple domains.
- Demonstrate adaptability.

---

# 4. Dataset Split

- Train: 70%
- Validation: 15%
- Test: 15%

Use stratified splitting where possible.

---

# 5. Data Preparation

## Shared
- Remove duplicates
- Handle missing values
- Preserve original text
- Standardize labels
- Create reproducible splits

## Classical Model
- Lowercase
- TF-IDF vectorization
- Preserve negation
- Optional lemmatization

## DistilBERT
- Minimal cleaning
- Hugging Face tokenizer
- Truncation
- Padding

---

# 6. Models

## Baseline
- TF-IDF
- Logistic Regression

## Transformer
- DistilBERT

---

# 7. Topic Modeling

Implement one method only.

Preferred order:
1. NMF
2. BERTopic (only if time allows)

Evaluate:
- Topic coherence
- Interpretability
- Representative reviews

---

# 8. Aspect-Based Sentiment

Use a rule-based aspect extractor.

Evaluate:
- Correct aspect detection
- Aspect frequency
- Sentiment aggregation
- Manual inspection

Known limitation:
Review-level sentiment is reused for detected aspects.

---

# 9. Evaluation Metrics

## Sentiment
- Accuracy
- Precision
- Recall
- Macro F1
- Weighted F1
- Confusion Matrix

## Runtime
- Training time
- Inference latency
- Model size

## Topic Modeling
- Human interpretability
- Topic coherence (if practical)
- Topic coverage

---

# 10. Error Analysis

Inspect failures involving:
- Sarcasm
- Mixed sentiment
- Very short reviews
- Very long reviews
- Domain-specific terminology
- Rating/text disagreement

---

# 11. Cross-Domain Validation

Run the full pipeline on at least two datasets and compare:
- Model performance
- Topic quality
- Aspect usefulness
- Preprocessing adjustments

---

# 12. Result Tables

## Sentiment Models

| Metric | Logistic Regression | DistilBERT |
|---|---:|---:|
| Accuracy | | |
| Precision | | |
| Recall | | |
| Macro F1 | | |
| Weighted F1 | | |
| Inference Time | | |

## Topic Modeling

| Metric | Result |
|---|---|
| Number of Topics | |
| Average Topic Size | |
| Representative Topic | |
| Notes | |

---

# 13. Success Criteria

The experiments are successful if:
- DistilBERT matches or exceeds the baseline.
- Topics are understandable.
- Aspect summaries provide useful insights.
- The pipeline performs reasonably across at least two datasets.
- Results are reproducible.

---

# 14. Known Limitations

- Rule-based aspects may miss unseen terminology.
- Topic quality depends on the dataset.
- DistilBERT may not generalize equally across domains.
- Business insights are template-based.

---

# 15. Future Work (Outside MVP)

- Multiple transformer models
- Statistical significance testing
- MLflow or Weights & Biases
- Ablation studies
- SHAP/LIME
- Learned aspect extraction
- LLM-generated summaries

---

# 16. Deliverables

The repository should include:
- Trained Logistic Regression baseline
- DistilBERT evaluation
- Topic modeling results
- Aspect analysis examples
- Error analysis
- Performance comparison tables
- Figures for the README
