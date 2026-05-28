# ML Engineering

End-to-end machine learning engineering and MLOps workflows built during the
appliedAI ML & MLOps track.

This repository documents the progression through an intensive 6-week
industry-oriented specialization program focused on practical machine learning
engineering, workflow orchestration, reproducibility, and MLOps fundamentals.
The program combines weekly project work, mentor evaluations, and hands-on
implementation of real-world machine learning systems.

The track is structured around progressively building complete ML workflows —
from exploratory analysis and preprocessing to model training, orchestration,
experiment tracking, and deployment-oriented workflows.

---

## Track Structure

### Week 1 — Classification Model / ML Foundations

Introduction to supervised machine learning using the Titanic dataset.

Topics covered:

- exploratory data analysis
- preprocessing and feature scaling
- logistic regression with scikit-learn
- logistic regression from scratch using NumPy
- gradient descent optimization
- custom evaluation metrics
- ROC/AUC evaluation
- automated testing and CI integration

---

### Weeks 2–4 — Bike Rental Forecasting Project

End-to-end machine learning engineering project focused on forecasting
city-wide bike rental demand.

Topics covered:

- data preprocessing pipelines
- dataset validation and quarantine handling
- feature engineering
- workflow orchestration with Dagster
- dataset generation and persistence
- forecasting workflows
- experiment tracking
- reproducibility and MLOps foundations

---

### Weeks 5–6 — Industry Challenge

Team-based machine learning challenge in collaboration with an industry
partner.

Challenge focus:

> Design, build, and present an end-to-end machine learning solution that
> unlocks insights hidden in polymer material data.

The final challenge will combine the concepts learned throughout the track into
a complete ML engineering workflow including data preparation, model
development, experimentation, and presentation.

---

## Current Progress

### 00 — Classification Model

- [x] Repository setup
- [x] Python project configuration
- [x] Exploratory data analysis
- [x] Logistic regression with scikit-learn
- [x] Logistic regression from scratch using NumPy
- [x] Custom evaluation metrics
- [x] ROC AUC evaluation
- [x] Automated unit tests
- [x] CI integration
- [x] Model comparison and parameter analysis

### 01 — Bike Rental Forecasting

- [x] Initial project structure
- [x] Exploratory data analysis
- [x] Preprocessing workflow design
- [x] Dagster asset pipeline
- [x] Structured dataset validation
- [x] Quarantine handling
- [x] Feature engineering
- [x] Dataset materialization
- [x] Automated tests
- [x] CI integration
- [ ] Forecasting model training
- [ ] MLflow integration
- [ ] Deployment-oriented workflows

### 02 — Industry Challenge

- [ ] Upcoming

---

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Dagster
- MLflow
- Jupyter Notebooks
- Ruff
- pytest
- uv
- GitHub Actions

---

## Repository Structure

```text
.
├── 00_classification_model/
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── notebooks/
│   ├── reports/
│   ├── src/
│   └── tests/
│
├── 01_bike_rental_forecasting/
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── data/
│   ├── notebooks/
│   ├── reports/
│   ├── src/
│   └── tests/
│
└── README.md
```

---


## Goals

This repository focuses on building practical machine learning engineering
skills, including:

- understanding machine learning fundamentals
- building reproducible preprocessing workflows
- implementing ML models from scratch
- structuring maintainable ML codebases
- applying workflow orchestration
- integrating testing and CI/CD
- learning production-oriented MLOps practices

---

## Quickstart

Each project uses `uv` together with a dedicated `pyproject.toml` and
`uv.lock` file for isolated dependency and environment management.

### Classification Model

```bash
cd 00_classification_model

uv sync --dev
```

### Bike Rental Forecasting

```bash
cd 01_bike_rental_forecasting

uv sync --dev
uv run dg dev
```