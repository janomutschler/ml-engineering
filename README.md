# ml-engineering

End-to-end machine learning engineering and MLOps workflows built during the appliedAI ML & MLOps Track (ongoing).

The repository evolves throughout the track and covers topics ranging from exploratory data analysis and model training to reproducible workflows, testing, and deployment-oriented ML systems.

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

### Week 2
- [ ] Upcoming
---

## Repository Structure

- `data/raw/` — source datasets used throughout the exercises
- `notebooks/` — exploratory analysis and model development notebooks
- `src/` — reusable Python modules and utilities
- `subjects/` — assignment descriptions and requirements
- `reports/` — written summaries and exported figures
- `tests/` — automated tests and validation logic

---

## Current Focus

Completed Week 1 on binary classification and logistic regression, including a full NumPy implementation built from scratch.

Currently starting Week 2 with a focus on machine learning data pipelines and workflow orchestration.

---

## Quickstart

This project uses `uv` together with `pyproject.toml` for dependency and environment management.

### Setup

```bash
# install dependencies and create local environment
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