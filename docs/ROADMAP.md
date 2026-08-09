# ROADMAP
## AI Customer Feedback Intelligence Platform — MVP

**Goal:** Build and deploy a strong portfolio-ready MVP that demonstrates practical NLP, machine learning, experimentation, and dashboard development without expanding into unnecessary production features.

## MVP Definition

The MVP is complete when a user can:

1. Upload a CSV containing customer feedback.
2. Select or confirm the text column.
3. Run sentiment analysis.
4. Discover major feedback topics.
5. View aspect-level sentiment.
6. See an executive summary and key business insights.
7. Explore results in an interactive dashboard.
8. Export the enriched dataset.

The MVP should prioritize a reliable end-to-end workflow over advanced platform features.

## Phase 0 — Project Setup

### Objectives
- Create the repository.
- Set up the Python environment.
- Define the folder structure.
- Add basic project documentation.
- Select the first development dataset.

### Deliverables
- GitHub repository
- `requirements.txt` or `pyproject.toml`
- `.gitignore`
- Initial folder structure
- Working Streamlit entry point
- Sample dataset stored outside the main source code

### Completion Criteria
- The application launches locally.
- Dependencies install from a clean environment.
- The repository is ready for feature development.

## Phase 1 — Data Ingestion and Validation

### Objectives
Build a reusable upload and validation workflow for customer-feedback CSV files.

### Features
- CSV upload
- Dataset preview
- Text-column detection
- Manual text-column override
- Basic schema validation
- Missing-value handling
- Duplicate detection
- File-size and row-count checks
- Canonical internal column names

### Deliverables
- Reusable ingestion module
- Dataset validation messages
- Cleaned internal DataFrame
- Unit tests for critical ingestion logic

### Completion Criteria
- A user can upload a valid CSV without editing code.
- The system correctly identifies or accepts the review-text column.
- Invalid files produce clear error messages.

## Phase 2 — Exploratory Data Analysis

### Objectives
Create a compact analysis layer that helps users understand the dataset before running advanced NLP.

### Features
- Number of reviews
- Missing-value summary
- Review-length distribution
- Rating distribution, when available
- Reviews over time, when a date column exists
- Most frequent terms
- Basic dataset filters

### Deliverables
- EDA utility functions
- Initial overview dashboard
- Dataset summary cards

### Completion Criteria
- The dashboard adapts when optional columns are absent.
- Core dataset characteristics are visible before model inference.

## Phase 3 — Sentiment Analysis

### Objectives
Implement and compare one classical baseline with one transformer-based model.

### Required Models
- TF-IDF + Logistic Regression
- DistilBERT sentiment model

### Features
- Positive, neutral, and negative labels
- Prediction confidence where supported
- Sentiment distribution
- Sentiment trend over time
- Representative reviews by sentiment
- Clearly defined default model

### Deliverables
- Baseline training pipeline
- Saved vectorizer and Logistic Regression model
- DistilBERT inference pipeline
- Model comparison table
- Sentiment dashboard page

### Completion Criteria
- Both models generate predictions on the evaluation dataset.
- Core evaluation metrics are recorded.
- Sentiment results appear correctly in the dashboard.

## Phase 4 — Topic Modeling

### Objectives
Identify the main themes discussed in customer feedback.

### MVP Approach
Use one practical method:
- NMF with TF-IDF, or
- BERTopic if implementation remains manageable

Choose the method that produces the clearest topics within the project timeline.

### Features
- Configurable number of topics
- Topic keywords
- Topic frequency
- Average sentiment by topic
- Representative reviews

### Deliverables
- Topic-modeling module
- Topic assignment for each review
- Topics dashboard page

### Completion Criteria
- Topics are interpretable on the development dataset.
- Users can inspect representative reviews for each topic.

## Phase 5 — Aspect-Based Sentiment

### Objectives
Show what customers are discussing and how they feel about each aspect.

### MVP Approach
Use a practical rule-based or keyword-based aspect system.

Example aspects:
- Price
- Quality
- Performance
- Shipping
- Customer service
- Usability

### Features
- Aspect detection
- Sentiment by aspect
- Aspect frequency
- Most positive and negative aspects
- Evidence reviews

### Deliverables
- Aspect configuration file
- Aspect extraction module
- Aspect-sentiment summary
- Aspect Analysis dashboard page

### Completion Criteria
- The system extracts meaningful aspects from the primary dataset.
- Every displayed aspect insight links to example reviews.
- No large custom aspect-model training pipeline is required.

## Phase 6 — Summaries and Business Insights

### Objectives
Convert model outputs into concise, decision-oriented findings.

### Features
- Executive summary
- Top complaints
- Top praised areas
- Most discussed topics
- Worst-performing aspects
- Emerging issue indicators when date data is available
- Suggested areas for investigation

### MVP Constraints
- Insights must come from measurable results.
- Recommendations must remain cautious and evidence-based.
- A conversational LLM assistant is not required.

### Deliverables
- Insight-generation module
- Summary cards
- Evidence links
- Insights dashboard page

### Completion Criteria
- The platform generates a useful summary.
- Insights are traceable to reviews and metrics.
- Unsupported claims are avoided.

## Phase 7 — Dashboard Integration

### Objectives
Combine all completed modules into one coherent Streamlit application.

### MVP Pages
1. Upload & Setup
2. Overview
3. Sentiment
4. Topics
5. Aspect Analysis
6. Insights
7. Data Explorer

### Features
- Shared filters
- Session-state management
- Loading indicators
- Empty states
- Error handling
- Responsive chart sizing
- CSV export

### Deliverables
- Complete multi-page Streamlit application
- Consistent navigation
- Reusable dashboard components
- End-to-end analysis workflow

### Completion Criteria
- A new user can complete the analysis without reading source code.
- Filters behave consistently.
- The app handles missing optional columns safely.

## Phase 8 — Evaluation, Testing, and Polish

### Objectives
Make the project reliable and presentable enough for recruiters and technical interviews.

### Tasks
- Complete model evaluation
- Perform targeted error analysis
- Add unit tests for critical modules
- Improve exception handling
- Review chart labels and titles
- Remove dead code
- Add type hints and concise docstrings
- Test with at least two datasets
- Record known limitations

### Deliverables
- Final metrics table
- Error-analysis examples
- Test suite for core logic
- Clean repository
- Screenshots or short demo GIF

### Completion Criteria
- The primary workflow is stable.
- Known limitations are documented.
- The project can be demonstrated in a few minutes.

## Phase 9 — Deployment and Portfolio Release

### Objectives
Publish the MVP and make it easy to evaluate.

### Tasks
- Deploy the Streamlit application
- Finalize `README.md`
- Add setup instructions
- Add architecture diagram
- Add model results
- Include screenshots or a demo GIF
- Verify links
- Create a tagged MVP release

### Deliverables
- Public GitHub repository
- Live application link or reliable local setup
- Final README
- MVP release tag
- Resume-ready project description

### Completion Criteria
- A recruiter can understand the problem, solution, and results from the README.
- The project is ready for a resume, LinkedIn, and interviews.

## Suggested Build Order

```text
Project Setup
    ↓
Data Ingestion
    ↓
EDA
    ↓
Sentiment Baseline
    ↓
DistilBERT
    ↓
Topic Modeling
    ↓
Aspect Sentiment
    ↓
Insights
    ↓
Dashboard Integration
    ↓
Testing and Deployment
```

## Priority Levels

### Must Have
- CSV upload
- Text-column selection
- Data validation
- Logistic Regression baseline
- DistilBERT sentiment analysis
- Topic modeling
- Aspect-level sentiment
- Executive summary
- Interactive Streamlit dashboard
- Export
- Model evaluation
- Deployment or reproducible local setup

### Nice to Have
- Automatic domain detection
- Custom topic labels
- Emerging issue detection
- Confidence filtering
- Dark mode
- Additional datasets
- Advanced explainability

### Explicitly Outside the MVP
- Authentication
- Billing
- Team workspaces
- Persistent cloud storage
- Real-time streaming
- FastAPI microservices
- Kubernetes
- Model registry
- Multilingual support
- LLM chat assistant
- Scheduled reports
- Enterprise monitoring
- Complex role-based permissions

## Time-Saving Rules

- Use one strong implementation per NLP capability.
- Avoid infrastructure that does not improve the demo directly.
- Do not build both NMF and BERTopic unless time remains after deployment.
- Use a rule-based aspect system before considering a trained aspect model.
- Prioritize a polished end-to-end workflow over additional algorithms.
- Stop adding features once the MVP completion criteria are satisfied.

## Final Definition of Done

The project is portfolio-ready when:

- A user can upload and analyze a new CSV.
- Sentiment, topics, and aspects are generated successfully.
- The dashboard communicates clear, evidence-backed insights.
- Classical and transformer sentiment models are compared.
- Core results are reproducible.
- The repository is clean and documented.
- The application has a working demo or clear local setup.
- The project can be explained confidently in a five-minute walkthrough.
