
# DATA_INGESTION_PHASE.md

# Data Ingestion Phase

## Purpose

This phase implements the data ingestion pipeline that transforms uploaded CSV files into a clean, validated, canonical DataFrame. Every downstream component (EDA, sentiment analysis, topic modeling, aspect analysis, and insights) depends on the quality of this phase.

This implementation follows the requirements defined in **ROADMAP.md** and **TECHNICAL_DESIGN.md**.

---

# Objectives

- Accept customer feedback CSV files
- Detect the review text column automatically
- Allow manual column selection
- Validate dataset quality
- Handle missing and duplicate data
- Normalize column names
- Detect optional metadata columns
- Produce a canonical DataFrame

---

# Deliverables

- CSV upload module
- Dataset validation module
- Column detection utility
- Canonical DataFrame generator
- Error handling
- Unit tests

---

# Recommended Implementation Order

1. Create ingestion package
2. Build CSV loader
3. Build schema validator
4. Implement automatic text-column detection
5. Implement metadata detection
6. Clean and normalize data
7. Generate canonical DataFrame
8. Integrate with Streamlit upload page
9. Write tests

---

# Files to Create

```text
src/
└── ingestion/
    ├── __init__.py
    ├── loader.py
    ├── validator.py
    ├── schema.py
    ├── column_detector.py
    └── cleaning.py
```

## File Responsibilities

**loader.py**
- Read CSV files
- Handle encoding issues
- Return a pandas DataFrame

**validator.py**
- Validate uploaded files
- Generate user-friendly validation messages

**schema.py**
- Define required and optional columns
- Store canonical column names

**column_detector.py**
- Detect the most likely review text column
- Detect optional metadata columns

**cleaning.py**
- Remove empty reviews
- Remove duplicates
- Normalize column names
- Create canonical DataFrame

---

# Upload Workflow

```text
User Upload
      ↓
Read CSV
      ↓
Validate File
      ↓
Detect Text Column
      ↓
Detect Optional Columns
      ↓
Clean Dataset
      ↓
Canonical DataFrame
      ↓
Return to Application
```

---

# Automatic Text Column Detection

Candidate columns should be ranked using:

- String data type
- Average text length
- Percentage of populated values
- Column names containing keywords such as:
  - review
  - feedback
  - comment
  - text
  - description
  - message

The highest-scoring column should be suggested, while allowing the user to override it.

---

# Optional Metadata Detection

Attempt to identify:

| Internal Name | Possible Source Columns |
|---|---|
| date | date, review_date, created_at |
| rating | rating, score, stars |
| product | product, game, app, service |
| category | category, department |
| version | version, release |

Missing optional columns should never stop the pipeline.

---

# Validation Rules

Reject or warn when:

- File is not CSV
- Dataset is empty
- No usable text column exists
- Selected text column is empty
- Encoding cannot be read
- File exceeds supported size

Warnings should explain how the user can fix the issue.

---

# Canonical DataFrame

Required internal columns:

- review_id
- review_text
- clean_text

Optional columns:

- date
- rating
- product
- category
- version

All downstream modules should rely only on these canonical names.

---

# Streamlit Integration

The Upload page should provide:

- File uploader
- Dataset preview
- Suggested text column
- Manual override
- Validation messages
- "Run Analysis" button

The analysis pipeline should not begin until validation succeeds.

---

# Testing

## Unit Tests

- CSV loading
- Invalid file handling
- Text-column detection
- Duplicate removal
- Missing-value handling
- Canonical schema creation

## Integration Tests

- Upload valid dataset
- Upload invalid dataset
- Upload dataset with optional metadata
- Upload text-only dataset

---

# Suggested Git Commits

- Create ingestion package
- Implement CSV loader
- Add validation
- Add column detection
- Generate canonical DataFrame
- Integrate upload page
- Add ingestion tests

---

# Common Pitfalls

- Assuming fixed column names
- Modifying original review text
- Hardcoding file paths
- Rejecting datasets because optional columns are missing
- Failing to display helpful validation messages

---

# Definition of Done

This phase is complete when:

- A user can upload a supported CSV.
- The application identifies the review text column.
- Optional metadata is detected when available.
- Invalid datasets return clear errors.
- A canonical DataFrame is produced.
- Unit and integration tests pass.
- The output is ready for the EDA phase.
