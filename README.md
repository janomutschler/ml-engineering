# ml-engineering

End-to-end machine learning engineering and MLOps workflows built during the appliedAI ML & MLOps Track (ongoing).

The repository evolves throughout the track and covers topics ranging from exploratory data analysis and model training to reproducible ML pipelines and workflow orchestration.

---

## Current Progress

### Week 1 — Binary Classification
- [x] Repository setup
- [x] Python project configuration and CI
- [x] Titanic exploratory data analysis
- [x] Scikit-learn logistic regression baseline
- [x] Logistic regression from scratch using NumPy
- [x] Custom evaluation metrics and confusion matrix
- [x] Automated unit tests
- [x] CI workflow

### Week 2 — Data Pipeline
- [x] Project structure initialization
- [x] Dagster dependencies setup
- [ ] Bike rental preprocessing pipeline
- [ ] Feature engineering
- [ ] Workflow orchestration with Dagster

---

## Repository Structure

- `data/` — raw and processed datasets
- `docs/` — weekly summaries and documentation
- `notebooks/` — exploratory analysis and development notebooks
- `src/` — reusable Python modules and pipeline logic
- `subjects/` — assignment descriptions and requirements
- `tests/` — automated tests and validation logic

---

## Current Focus

Week 1 established the foundations of supervised machine learning using logistic regression and a custom NumPy implementation.

Week 2 focuses on building a reusable preprocessing pipeline for a bike-rental forecasting project using pandas and Dagster.

---

## Quickstart

This project uses `uv` together with `pyproject.toml` for dependency and environment management.

### Setup

```bash
uv sync --dev
```

---

## Week 1 Contents

### Notebooks
- `01_eda.ipynb` — exploratory data analysis and preprocessing exploration
- `02_logreg_sklearn.ipynb` — logistic regression baseline using scikit-learn
- `03_logreg_numpy.ipynb` — full logistic regression implementation from scratch using NumPy

### Core Concepts Covered
- binary classification
- train/test splitting
- feature standardization
- logistic regression
- sigmoid activation
- binary cross-entropy loss
- gradient descent optimization
- custom evaluation metrics
- automated testing