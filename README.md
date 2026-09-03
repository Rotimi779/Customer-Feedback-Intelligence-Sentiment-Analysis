# AI Customer Feedback Intelligence Platform

A portfolio-focused NLP application that converts customer-feedback CSV files
into a standardized analysis dataset, explores it through EDA, and enriches each
review with sentiment, topic, and rule-based aspect outputs before the later
business-insights and final dashboard phases.

## Current status

The cumulative codebase now includes:

1. Project Setup
2. Data Ingestion and Validation
3. Exploratory Data Analysis
4. Sentiment Analysis
5. Topic Modeling
6. Aspect Analysis

Phase 6 adds a transparent rule-based aspect extractor, synonym/phrase mapping,
review-sentiment association, aspect aggregation, visualizations, Streamlit
exploration, and tests. The MVP intentionally reuses review-level sentiment for
each detected aspect; it does not claim clause-level aspect sentiment.

## Application workflow

1. Open **Upload & Setup**.
2. Upload a CSV or select the bundled sample dataset.
3. Confirm review-text and optional metadata mappings.
4. Prepare the canonical dataset.
5. Open **Overview** for model-free EDA and filtering.
6. After local sentiment artifacts exist, open **Sentiment**.
7. Select a trained model and run sentiment inference.
8. Open **Topics**, choose a topic count, and run NMF topic modeling.
9. Open **Aspect Analysis** and run the rule-based aspect stage.
10. Inspect aspect frequency, sentiment, optional rating rankings, associated topics,
    and supporting reviews.

Uploaded customer files are processed in memory. Do not upload confidential or
sensitive personal data to this portfolio application.

## Requirements

- Python 3.11+
- Git

The checked-in `requirements.txt` includes Streamlit, pandas, NumPy,
scikit-learn, Transformers, PyTorch, Plotly, joblib, and pytest.

## Local setup

```bash
python -m venv .venv
```

Activate the environment.

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

Then:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest
streamlit run app.py
```

## Canonical data contract

Required internal columns produced by ingestion:

- `review_id`
- `review_text`
- `clean_text`

Optional normalized metadata:

- `date`
- `rating`
- `product`
- `category`
- `version`

Phase 4 appends:

- `sentiment_label` — `Negative`, `Neutral`, or `Positive`
- `sentiment_score` — confidence for the predicted class

Phase 5 appends:

- `topic_id`
- `topic_label`

Phase 6 appends:

- `detected_aspects` — zero or more canonical aspect names
- `aspect_sentiment` — aspect-to-review-sentiment mapping
- `aspect_confidence` — optional aspect-to-review-confidence mapping

Each stage returns a copy rather than mutating its input DataFrame.

## Phase 4 training data

Training/evaluation data must contain a review-text column and a labelled
three-class sentiment column. By default the scripts expect:

```text
review_text
sentiment_label
```

The label column must resolve to `Negative`, `Neutral`, and `Positive`.
Numeric or domain-specific labels are never guessed automatically. Supply an
explicit JSON map when needed, for example:

```powershell
--label-map-json '{"0":"Negative","1":"Neutral","2":"Positive"}'
```

The experiment code removes empty reviews and duplicates, standardizes labels,
and creates deterministic stratified **70/15/15** train/validation/test splits.
Both model-training scripts use the same split implementation and default seed.

## Train the baseline

Place a labelled dataset under `data/training/` (or point to another local CSV):

```powershell
python -m src.sentiment.train_baseline `
  --input data/training/YOUR_LABELLED_DATA.csv `
  --dataset-name primary_sentiment_dataset
```

Artifacts are written to:

```text
models/logistic_regression/
├── tfidf_vectorizer.joblib
├── logistic_regression.joblib
├── metadata.json
├── metrics.json
└── test_predictions.json
```

## Fine-tune DistilBERT

```powershell
python -m src.sentiment.train_transformer `
  --input data/training/YOUR_LABELLED_DATA.csv `
  --dataset-name primary_sentiment_dataset
```

The default base checkpoint is DistilBERT uncased. The training loop uses
minimal transformer preprocessing, mini-batches, AdamW, a linear warmup
schedule, validation Macro F1, and early stopping. The best checkpoint is saved
locally under `models/distilbert/` with its tokenizer, evaluation report,
training metadata, and test predictions.

DistilBERT training is substantially heavier than Logistic Regression. CPU
training is supported by the code but a CUDA-capable environment is much
faster when available.

## Compare and select the production model

Print the common evaluation table:

```powershell
python -m src.sentiment.evaluation
```

The table uses the metrics required by the experiment plan:

- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Weighted F1
- Training time
- Inference time / throughput
- Model size
- Confusion matrix (stored in each report)

After reviewing performance and practical inference speed, record the production
choice explicitly rather than letting the application guess:

```powershell
python -m src.sentiment.evaluation `
  --production-model distilbert `
  --rationale "Higher Macro F1 with acceptable inference speed on the evaluation set."
```

This creates `models/production_model.json`. The Sentiment page uses that model
as the preferred default when its artifacts are present.

## Sentiment architecture

```text
Canonical DataFrame
        |
        +--> clean_text --> TF-IDF --> Logistic Regression
        |
        +--> review_text --> DistilBERT tokenizer --> DistilBERT
                                      |
                                      v
                         sentiment_label + score
                                      |
                                      v
                             Enriched DataFrame
```

Training code and inference code are intentionally separated. The Streamlit app
uses only the stable `SentimentAnalyzer` interface and does not know how either
model was trained.

## Repository structure (Phase 4)

```text
customer-feedback-intelligence/
├── app.py
├── pages/
│   ├── 1_Overview.py
│   ├── 2_Sentiment.py
│   └── ...
├── src/
│   ├── ingestion/
│   ├── eda/
│   ├── preprocessing/
│   │   ├── cleaner.py
│   │   └── text_utils.py
│   ├── sentiment/
│   │   ├── baseline.py
│   │   ├── transformer.py
│   │   ├── preprocessing.py
│   │   ├── train_baseline.py
│   │   ├── train_transformer.py
│   │   ├── inference.py
│   │   ├── evaluation.py
│   │   ├── metrics.py
│   │   └── utils.py
│   ├── topics/
│   ├── aspects/
│   ├── insights/
│   └── pipeline.py
├── models/
│   ├── logistic_regression/
│   └── distilbert/
├── data/
│   ├── sample/
│   └── training/
├── tests/
├── docs/
└── requirements.txt
```

## MVP boundaries

The project continues to exclude authentication, billing, persistent cloud
storage, microservices, real-time streaming, multilingual product support,
enterprise monitoring, and an LLM chat assistant.

# Phase 5 — Topic Modeling

Phase 5 adds one transparent topic-modeling method to the sentiment-enriched
review pipeline: **TF-IDF + Non-negative Matrix Factorization (NMF)**. BERTopic
is intentionally not included in the MVP.

The topic stage consumes the Phase 4 enriched DataFrame and appends:

```text
topic_id
topic_label
```

Topic labels are generated from the highest-weighted NMF keywords rather than
from an LLM. The model also produces reusable topic summaries containing topic
frequency, dataset share, keywords, representative reviews, and sentiment by
topic when sentiment predictions are available.

## Run topic modeling in Streamlit

1. Prepare a CSV on **Upload & Setup**.
2. Run sentiment inference on **Sentiment**.
3. Open **Topics**.
4. Choose a topic count from 5–15.
5. Run topic modeling.

The Topics page shows topic frequency, topic distribution, top keywords,
sentiment within topics, representative reviews, filters, and the quantitative
quality indicators used by the experiment plan.

## Offline topic-model experiment

The saved-model requirement is supported through the module CLI. For the
primary public dataset:

```powershell
python -m src.topics.modeling `
  --input data/training/primary_amazon_reviews_sentiment.csv `
  --dataset-name amazon_reviews `
  --n-topics 8
```

For the secondary domain:

```powershell
python -m src.topics.modeling `
  --input data/training/secondary_twitter_airline_sentiment.csv `
  --dataset-name twitter_airline `
  --n-topics 8
```

Each run writes a dataset-specific folder under `models/topic_model/` containing
NMF/vectorizer artifacts, topic metadata, topic assignments, a topic summary,
and evaluation metrics.

The evaluation reports:

- NPMI topic coherence
- topic diversity
- topic coverage
- repeat-run keyword stability
- average topic size
- whether all modeled reviews received a topic

Quantitative metrics do not replace manual interpretation. Review the topic
keywords and representative reviews before choosing the final topic count.

## Topic-model architecture

```text
Sentiment-enriched DataFrame
        |
        v
Topic-specific preprocessing
        |
        v
TF-IDF vectorizer
        |
        v
NMF
        |
        +--> topic keywords --> topic labels
        |
        +--> topic assignment per review
        |
        v
topic_id + topic_label
        |
        v
Topic summaries / visualizations
```

# Phase 6 — Aspect Analysis

Phase 6 uses a deterministic rule-based extractor built from canonical aspect
names, keywords, synonyms, and simple phrases. A review may match multiple
aspects. The extractor is modular: a different vocabulary can be injected for a
domain without replacing the matching algorithm.

The Streamlit workflow consumes Phase 5 topic-enriched sentiment results and
keeps `topic_id` / `topic_label` attached to every review so aspect evidence can
be inspected in its broader topic context.

The MVP reuses the existing review-level `sentiment_label` and
`sentiment_score` for every detected aspect. This known limitation is surfaced
in the UI and evaluation notes rather than being presented as a separate trained
aspect-sentiment model.

Aspect aggregation provides:

- mention frequency and unique-review coverage
- average/dominant sentiment
- positive, neutral, and negative counts/shares
- average confidence when available
- average rating when rating metadata exists
- supporting reviews and associated topic context

Automated evaluation reports structural coverage and sentiment-association
completeness. Correct extraction, label relevance, and business usefulness still
require manual inspection because the MVP does not include a gold aspect-labelled
dataset.
