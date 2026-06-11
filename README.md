# Machine Learning Engineering & MLOps

A portfolio of machine learning engineering and MLOps projects, progressing from fundamentals to a full production-style ML system. Built during the appliedAI ML & MLOps track — a 6-week, industry-oriented specialization in practical ML engineering, orchestration, reproducibility, and MLOps.

The centerpiece is **01 — Bike Rental Demand Forecasting**, an end-to-end MLOps pipeline. **00** is a from-first-principles fundamentals warm-up; **02** is an upcoming team industry challenge.

---

## Featured — Bike Rental Demand Forecasting (Weeks 2–4)

> A production-style system that forecasts **day-ahead hourly bike-rental demand** for a city-wide bike-sharing service — covering the full lifecycle from raw data to a served, versioned, fully-traceable model.

**Stack:** Python · pandas · scikit-learn · XGBoost · LightGBM · Dagster · MLflow · LakeFS · FastAPI · Docker · Pytest · Ruff · uv

- **Orchestrated lifecycle** as a Dagster asset graph: ingestion → feature engineering → evaluation → model registry → promotion → serving.
- **Honest time-series evaluation** via walk-forward backtesting (steady-state R² ≈ 0.90) instead of a single lucky split.
- **Automated model governance**: champion/challenger promotion through the MLflow registry — deploying a new model is an alias move, not a redeploy.
- **Reproducibility & lineage**: every registered model traces back to its MLflow run, git commit, and the exact LakeFS data snapshot it trained on.
- **No training–serving skew**: the prediction API computes features with the *same* code used in training, verified by tests.

→ Full write-up, architecture, and results: [`01_bike_rental_forecasting/README.md`](01_bike_rental_forecasting/README.md)

---

## All projects

### 00 — Classification Model (Week 1)

A compact, from-first-principles introduction to supervised learning on the Titanic dataset: logistic regression both with scikit-learn and implemented from scratch in NumPy (gradient descent), plus custom evaluation metrics, ROC/AUC, and automated tests with CI. Deliberately small — the foundations warm-up before the main project.

→ [`00_classification_model/`](00_classification_model/)

### 01 — Bike Rental Demand Forecasting (Weeks 2–4)

The featured project above: a full Dagster + MLflow + LakeFS + FastAPI MLOps system for day-ahead demand forecasting. See its [README](01_bike_rental_forecasting/README.md) for architecture, results, and setup.

→ [`01_bike_rental_forecasting/`](01_bike_rental_forecasting/)

### 02 — Industry Challenge (Weeks 5–6) — *upcoming*

A team-based machine learning challenge with industry partner [moldflow.eu](https://moldflow.eu), working with polymer-material data. The goal is to design, build, and present an end-to-end ML solution that surfaces insights hidden in the data — combining the track's workflows into one applied, collaborative deliverable. *Details and code to be added as the challenge runs.*

---

## What this repository demonstrates

A deliberate progression from ML fundamentals to production MLOps:

- **Fundamentals** — implementing models from scratch, evaluation metrics, gradient descent *(00)*.
- **ML engineering** — reproducible pipelines, feature engineering, workflow orchestration, testing and CI *(01)*.
- **MLOps** — experiment tracking, a model registry with automated promotion, data versioning and end-to-end lineage, and model serving *(01)*.
- **Applied teamwork** — an industry challenge on real material data *(02)*.

---

## Technologies

- **Core:** Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM
- **MLOps & orchestration:** Dagster, MLflow, LakeFS, FastAPI, Docker
- **Tooling:** uv, Ruff, Pytest, GitHub Actions, Jupyter

---

## Repository structure

```text
.
├── 00_classification_model/      # Week 1   — fundamentals (Titanic, logistic regression)
├── 01_bike_rental_forecasting/   # Weeks 2–4 — end-to-end MLOps system (featured)
└── README.md                     # (02_industry_challenge added during Weeks 5–6)
```

Each project is self-contained, with its own `pyproject.toml` and `uv.lock` for isolated dependency and environment management.

---

## Quickstart

Each project manages its own environment with `uv`.

```bash
# 00 — Classification (fundamentals)
cd 00_classification_model && uv sync --dev

# 01 — Bike Rental (full stack: LakeFS + MLflow + Dagster + API)
cd 01_bike_rental_forecasting && make install
# then follow the project README: services start via `make infra`, `make mlflow`, `make dev`, `make api`
```