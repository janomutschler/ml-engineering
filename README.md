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
- [ ] Logistic regression from scratch using NumPy

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

Week 1 focuses on a Titanic survival classification problem using logistic regression.

The assignment consists of:
1. Building a baseline workflow using scikit-learn
2. Implementing logistic regression from scratch using NumPy

Future weeks will expand toward more advanced machine learning engineering and MLOps workflows.

---

## Quickstart

This project uses `uv` together with `pyproject.toml` for dependency and environment management.

### Setup

```bash
# install dependencies and create local environment
uv sync --dev