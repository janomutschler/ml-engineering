# Machine Learning Engineering & MLOps

A portfolio of machine learning engineering and MLOps projects, progressing from fundamentals to a production-style ML system and a real-world industry challenge. Built during the appliedAI ML & MLOps track — a 6-week, industry-oriented specialization in practical ML engineering, orchestration, reproducibility, and MLOps.

The centerpiece is **01 — Bike Rental Demand Forecasting**, an end-to-end MLOps pipeline. **00** is a compact fundamentals warm-up; **02** documents the final team-based industry challenge on material property prediction.

---

## Featured — Bike Rental Demand Forecasting (Weeks 2–4)

> A production-style system that forecasts **day-ahead hourly bike-rental demand** for a city-wide bike-sharing service — covering the full lifecycle from raw data to a served, versioned, fully traceable model.

**Stack:** Python · pandas · scikit-learn · XGBoost · LightGBM · Dagster · MLflow · LakeFS · FastAPI · Docker · Pytest · Ruff · uv

* **Orchestrated lifecycle** as a Dagster asset graph: ingestion → feature engineering → evaluation → model registry → promotion → serving.
* **Honest time-series evaluation** via walk-forward backtesting instead of a single lucky split.
* **Automated model governance** through the MLflow registry — deploying a new model is an alias move, not a redeploy.
* **Reproducibility & lineage**: every registered model traces back to its MLflow run, git commit, and the exact LakeFS data snapshot it trained on.
* **No training–serving skew**: the prediction API computes features with the same code used in training, verified by tests.

→ Full write-up, architecture, and results: [`01_bike_rental_forecasting/README.md`](01_bike_rental_forecasting/README.md)

---

## Applied Industry Challenge — Material Property Prediction (Weeks 5–6)

> A team-based industry challenge with [moldflow.eu](https://moldflow.eu), focused on predicting missing polymer material properties from anonymized industrial Moldflow data.

The task was to build a reproducible machine learning workflow for estimating ten anonymized material-property targets from material metadata, scalar parameters, data-quality indicators, and sequence-based curve data.

Because the dataset and implementation are tied to a private industry challenge, the public repository contains a documentation-only project summary. The full code, data, notebooks, and implementation details remain in a private repository due to data confidentiality and partner/IP constraints.

**What we covered:**

* Ingested and consolidated hundreds of material-level CSV files into a modeling dataset.
* Applied mandatory data-quality filters and investigated the optional high-quality `f012` filter.
* Built scalar-only baselines and target-specific feature-selection experiments.
* Engineered curve descriptors from sequence-based parameters `f017` and `f090`.
* Compared feature sets systematically: scalar-only, scalar + `f017`, scalar + `f090`, and combined curve/scalar setups.
* Evaluated multiple model families, including tree ensembles and gradient boosting models.
* Used repeated cross-validation with MAE, RMSE, R², and normalized MAE based on the central target range.
* Documented final model choices, feature importance findings, limitations, and future improvement ideas.

→ Public summary: [`02_industry_challenge/README.md`](02_industry_challenge/README.md)

---

## All projects

### 00 — Classification Model

A compact supervised-learning warm-up on the Titanic dataset: logistic regression with scikit-learn and a small from-scratch NumPy implementation, plus evaluation metrics and tests.

→ [`00_classification_model/`](00_classification_model/)

### 01 — Bike Rental Demand Forecasting

The featured project above: a full Dagster + MLflow + LakeFS + FastAPI MLOps system for day-ahead demand forecasting.

→ [`01_bike_rental_forecasting/`](01_bike_rental_forecasting/)

### 02 — Industry Challenge

A documentation-only public summary of the final team industry challenge on polymer material-property prediction. The implementation is kept private due to data confidentiality and partner/IP constraints.

→ [`02_industry_challenge/`](02_industry_challenge/)

---

## What this repository demonstrates

A deliberate progression from ML fundamentals to production-oriented MLOps and applied industry work:

* **Fundamentals** — model implementation, evaluation metrics, gradient descent, and testing.
* **ML engineering** — reproducible pipelines, feature engineering, workflow orchestration, CI, and structured project design.
* **MLOps** — experiment tracking, model registry workflows, data versioning, lineage, and model serving.
* **Applied industry ML** — working with anonymized real-world data, data-quality constraints, target-specific modeling, curve feature extraction, model comparison, and documentation of findings.
* **Collaboration** — team-based development, research notebooks, structured evaluation, and final presentation deliverables.

---

## Technologies

* **Core:** Python, pandas, NumPy, scikit-learn
* **Modeling:** XGBoost, LightGBM, CatBoost, ExtraTrees, Random Forest
* **MLOps & orchestration:** Dagster, MLflow, LakeFS, FastAPI, Docker
* **Tooling:** uv, Ruff, Pytest, GitHub Actions, Jupyter

---

## Repository structure

```text
.
├── 00_classification_model/       # Fundamentals warm-up
├── 01_bike_rental_forecasting/    # End-to-end MLOps system
├── 02_industry_challenge/         # Public summary of private industry challenge
└── README.md
```

Each implementation project is self-contained, with its own `pyproject.toml` and `uv.lock` for isolated dependency and environment management.

---

## Quickstart

Each implementation project manages its own environment with `uv`.

```bash
# 00 — Classification fundamentals
cd 00_classification_model && uv sync --dev

# 01 — Bike Rental MLOps system
cd 01_bike_rental_forecasting && make install
# then follow the project README:
# make infra, make mlflow, make dev, make api
```

The `02_industry_challenge` folder is documentation-only in this public repository. The full implementation is kept private due to data confidentiality and partner/IP constraints.
