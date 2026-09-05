# AI Customer Feedback Intelligence Platform

A portfolio-focused NLP application that converts customer-feedback CSV files
into a standardized analysis dataset, explores it through EDA, and enriches each
review with sentiment, topic, rule-based aspect outputs, and evidence-backed business
insights inside one integrated multi-page Streamlit dashboard.

## Current status

The cumulative codebase now includes:

1. Project Setup
2. Data Ingestion and Validation
3. Exploratory Data Analysis
4. Sentiment Analysis
5. Topic Modeling
6. Aspect Analysis
7. Business Insights
8. Dashboard Integration

Phase 8 connects the completed analysis modules through shared session state, global
filters, consistent UI helpers, a one-click full-analysis workflow, prerequisite/error
states, and a completed Data Explorer. It does not add a new ML model.

## Application workflow

1. Open **Upload & Setup**.
2. Upload a CSV or select the bundled sample dataset.
3. Confirm review-text and optional metadata mappings.
4. Confirm the canonical dataset.
5. Either click **Run Full Analysis** on Upload & Setup or rerun individual stages from
   their dedicated pages.
6. Open **Overview** for model-free EDA and shared filtering.
7. Explore **Sentiment**, **Topics**, **Aspect Analysis**, and **Insights**.
8. Global filters update saved results and summaries without rerunning ML models.
9. Open **Data Explorer** to inspect and download filtered enriched records.
10. Download the enriched CSV, Markdown insight report, or recommendation table.

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

The trained `models/distilbert/model.safetensors` file is intentionally kept as a
local artifact when it exceeds the repository hosting limit. Recreate it with the
training command above in a fresh clone; the application only offers DistilBERT
inference when the required local model weights are present.

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

## Repository structure (cumulative through Phase 8)

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
│   ├── dashboard/
│   │   ├── state.py
│   │   ├── filters.py
│   │   ├── components.py
│   │   ├── formatting.py
│   │   ├── errors.py
│   │   └── workflow.py
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

# Phase 7 — Business Insights

Phase 7 turns the outputs from Sentiment, Topics, and Aspect Analysis into a
decision-oriented layer without adding another machine-learning model. All summary
text and recommendations are deterministic and derived from measured results.

The insight engine reports at minimum:

- overall sentiment breakdown
- total reviews analyzed
- most discussed topic
- strongest supported positive aspect
- largest supported negative aspect / priority improvement area
- evidence-backed key findings
- cautious rule-based recommendations

When usable date metadata exists, it also calculates review volume, sentiment share,
topic prevalence, and aspect-negative-share trends. Recent change signals are shown
only when minimum evidence support is available, and the interface explicitly avoids
causal language.

The Insights page can export:

```text
customer_feedback_enriched.csv
business_insights.md
recommendations.csv
```

The enriched dataset remains the Phase 6 review-level schema; Phase 7 produces
separate structured insight objects rather than adding speculative text columns to
every review.

## Business-insight architecture

```text
Sentiment results
        |
Topic assignments
        |
Aspect results
        |
Optional metadata
        v
Deterministic insight generator
        |
        +--> Executive summary
        +--> Key findings + evidence
        +--> Rule-based recommendations
        +--> Optional trends
        v
Insights page + exports
```

Recommendations are intended as investigation priorities, not causal conclusions.
The known Phase 6 limitation also remains: review-level sentiment is reused for all
detected aspects in a review.



# Phase 8 — Dashboard Integration

Phase 8 keeps the Streamlit multi-page workflow and integrates the completed modules
into one application state. The dashboard now supports:

- shared session-state initialization and stale-output invalidation
- one **Run Full Analysis** action for sentiment → topics → aspects → insights
- adaptive global filters for date, product, category, rating, sentiment, topic, aspect, and text
- filter-only recomputation of summaries/visuals without rerunning ML
- consistent Positive/Neutral/Negative color semantics
- explicit page-prerequisite and empty-result states
- a completed Data Explorer with filtered enriched CSV export
- shared formatting and reusable dashboard components
- integrated dashboard regression tests

The existing per-stage buttons remain available for development and experimentation.
Running a stage manually invalidates only the downstream outputs that are no longer
current. Switching pages does not retrain or rerun saved model results.

The locally trained `models/distilbert/model.safetensors` file remains intentionally
excluded from Git when it exceeds repository hosting limits; Phase 8 does not replace
or regenerate local model artifacts.
