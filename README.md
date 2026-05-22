# ml-engineering

End-to-end machine learning engineering and MLOps workflows built during the appliedAI ML & MLOps Track (ongoing).

The repository evolves throughout the track and covers topics ranging from exploratory data analysis and model training to reproducible ML pipelines, workflow orchestration, and experiment management.

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
- [x] Bike rental preprocessing workflow exploration
- [x] Dataset aggregation and join validation
- [x] Temporal feature engineering prototype
- [ ] Dagster asset implementation
- [ ] Workflow orchestration with Dagster
- [ ] Automated dataset materialization

---

## Repository Structure

- `data/` — raw and processed datasets
- `docs/` — weekly summaries and documentation
- `notebooks/` — exploratory analysis and preprocessing notebooks
- `src/` — reusable Python modules and pipeline logic
- `subjects/` — assignment descriptions and requirements
- `tests/` — automated tests and validation logic

---

## Current Focus

Week 1 established the foundations of supervised machine learning using logistic regression and a custom NumPy implementation.

Week 2 focuses on designing and implementing a reusable preprocessing pipeline for a bike-rental forecasting project using pandas and Dagster.

---

## Quickstart

This project uses `uv` together with `pyproject.toml` for dependency and environment management.

### Setup

```bash
uv sync --dev