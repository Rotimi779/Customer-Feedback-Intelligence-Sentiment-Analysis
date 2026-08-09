
# PROJECT_SETUP_PHASE.md

# Project Setup Phase

## Purpose

This phase establishes the foundation for the AI Customer Feedback Intelligence Platform. By the end of this phase, the repository, development environment, project structure, and application skeleton should be ready so future phases can focus entirely on feature development.

This document follows the implementation order defined in `ROADMAP.md` and the architecture specified in `TECHNICAL_DESIGN.md`.

---

# Objectives

- Initialize the repository
- Configure the Python environment
- Create the project directory structure
- Install required dependencies
- Configure Git and `.gitignore`
- Build the initial Streamlit application
- Create placeholder modules
- Prepare sample datasets
- Verify the application launches successfully

---

# Deliverables

- Working Git repository
- Python virtual environment
- Installed dependencies
- Complete project folder structure
- Functional `app.py`
- Sample dataset directory
- Initial documentation
- Verified local setup

---

# Recommended Implementation Order

1. Initialize the repository
2. Create the directory structure
3. Create the virtual environment
4. Install dependencies
5. Configure Git
6. Build the Streamlit application skeleton
7. Create placeholder source modules
8. Add sample datasets
9. Verify the project runs

---

# Repository Structure

```text
customer-feedback-intelligence/
│
├── app.py
├── pages/
├── src/
├── models/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── tests/
├── docs/
├── README.md
├── ROADMAP.md
├── TECHNICAL_DESIGN.md
├── EXPERIMENTS.md
├── requirements.txt
└── .gitignore
```

Every directory should exist before feature development begins.

---

# Environment Setup

## Python

- Python 3.11 or newer

## Create Virtual Environment

```bash
python -m venv .venv
```

Activate the environment and upgrade pip before installing packages.

## Core Dependencies

- streamlit
- pandas
- numpy
- scikit-learn
- transformers
- torch
- plotly
- joblib
- pytest

Record all dependencies in `requirements.txt`.

---

# Configuration Files

Create:

- `.gitignore`
- `requirements.txt`
- `README.md`

The `.gitignore` should exclude virtual environments, caches, notebooks checkpoints, model artifacts (if appropriate), and Python bytecode.

---

# Streamlit Skeleton

Create `app.py` with:

- application title
- sidebar navigation
- placeholder welcome page
- logging initialization
- session state initialization

Verify the application launches:

```bash
streamlit run app.py
```

---

# Initial Source Modules

Create placeholder packages under `src/`:

- ingestion
- preprocessing
- sentiment
- topics
- aspects
- insights
- evaluation

Each package should contain an `__init__.py` file so imports work correctly.

---

# Dataset Preparation

Create:

```text
data/
├── raw/
├── processed/
└── sample/
```

Place at least one small public review dataset in `sample/` to support early development.

---

# Verification Checklist

Confirm:

- Repository initializes correctly
- Dependencies install without errors
- Streamlit launches
- Folder structure matches the technical design
- Sample dataset loads successfully
- Imports from `src` succeed

---

# Suggested Git Commits

- Initialize repository
- Create project structure
- Configure environment
- Add Streamlit skeleton
- Add sample dataset

---

# Common Pitfalls

- Mixing global and virtual Python environments
- Hardcoding file paths
- Missing `__init__.py` files
- Forgetting to pin dependency versions
- Placing datasets inside source code directories

---

# Definition of Done

This phase is complete when:

- Repository structure matches the technical design.
- The development environment is reproducible.
- Streamlit launches successfully.
- Core folders and placeholder modules exist.
- Sample datasets are available.
- The project is ready for the Data Ingestion phase.
