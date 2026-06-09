# Bike Rental Forecasting

End-to-end machine learning engineering project focused on preparing and forecasting city-wide bike rental demand.

This project is part of the appliedAI ML & MLOps track and evolves over multiple weeks from exploratory data analysis and preprocessing to model training, workflow orchestration, and MLOps workflows.

## Quick Start

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- make
- git

### Install

```bash
make install
```

**macOS only** — XGBoost requires `libomp` from Homebrew:
```bash
brew install libomp
```

### Run

The project requires two services running in parallel: an MLflow tracking server and the Dagster development server.

**Terminal 1 — MLflow tracking server:**

```bash
make mlflow
```

This starts the MLflow server at `http://127.0.0.1:5000` backed by a local SQLite database. The model registry requires a database-backed store; leave this running for the duration of your session.

**Terminal 2 — Dagster development server:**

```bash
make dev
```

This starts the Dagster UI at `http://127.0.0.1:3000` with `MLFLOW_TRACKING_URI` pre-configured. Materialize assets from the asset graph to run the pipeline end to end.

### Other commands

```bash
make test      # run the test suite
make lint      # check code style and formatting
make format    # auto-fix formatting and linting
```

## Project Goals

The project aims to:

* prepare operational bike rental data for machine learning
* build reproducible preprocessing pipelines
* engineer useful temporal and contextual features
* train forecasting models for bike rental demand
* integrate ML workflows into structured pipelines
* apply MLOps practices such as experiment tracking, model versioning, and deployment

## Current Scope

### Week 2 — Data Pipeline ✅

* source dataset ingestion through a reusable Dagster data loader resource
* custom Parquet IO manager for typed asset persistence
* source dataset schema validation with asset checks
* hourly rental activity aggregation
* weather anomaly correction and humidity imputation
* calendar feature generation
* unified bike rental feature dataset creation
* structured metadata tracking and logging
* reusable preprocessing transformations
* automated tests and CI validation

### Week 3 — EDA and Bike Rental Predictions ✅

* comprehensive exploratory data analysis
* data quality validation and consistency checks
* baseline forecasting models (Dummy, Linear Regression)
* cyclical temporal feature engineering
* historical demand lag features and contextual historical demand aggregations
* feature selection and evaluation
* Random Forest and XGBoost forecasting models
* feature importance analysis and model comparison

Results:

* XGBoost achieved an R² score above 0.918
* historical demand features provided the strongest predictive signal
* the complete forecasting workflow was integrated into Dagster

### ML Pipeline Refactor ✅

Production-readiness improvements applied after Week 3:

* **config-driven model selection** — model type and hyperparameters are configured via a typed `TrainingConfigResource`; each supported model has its own discoverable config block in the Dagster launchpad
* **split removed from the asset graph** — the chronological train/test split is computed in-process deterministically from a shared config, eliminating four persisted intermediate CSV assets
* **evaluation folded into training** — hold-out metrics (MAE, RMSE, RMSLE, R²) are attached as asset output metadata rather than persisted as a separate asset
* **LightGBM added** as a fifth model candidate alongside XGBoost, Random Forest, Linear Regression, and Dummy
* **Parquet persistence** for all intermediate DataFrame assets, eliminating datetime re-parsing machinery

### Phase 1 — MLflow Integration ✅

* every training run logged to MLflow with params, four hold-out metrics, model signature, input example, and provenance tags (Dagster run ID, git commit)
* trained model registered as a versioned artifact in the MLflow model registry (`bike_rental_forecaster`)
* `MlflowResource` owns the connection and run lifecycle; assets decide what to log
* tracking URI injected via `MLFLOW_TRACKING_URI` environment variable (fail-loud, no silent fallback)
* `mlflow.sklearn.log_model` used uniformly across all model types (XGBoost, LightGBM, Random Forest, and Linear Regression all expose the sklearn API)

## Planned Future Work

* walk-forward backtesting harness replacing the single holdout split
* champion/challenger promotion gate using MLflow model aliases
* data versioning with LakeFS
* end-to-end lineage: data commit → features → model → metrics
* batch inference and prediction asset
* input drift and performance decay monitoring
* daily schedule and file-arrival sensor
* deployment-oriented MLOps workflows

## Project Structure

```text
.
├── data/
│   ├── sources/         # raw input CSVs (not tracked by git)
│   ├── processed/       # materialized Dagster assets
│   └── quarantine/
├── mlartifacts/         # MLflow artifact store
├── mlflow.db            # MLflow tracking database (SQLite)
├── notebooks/
├── reports/
├── src/
│   └── bike_rental/
│       └── defs/
│           ├── assets/          # Dagster asset definitions
│           ├── asset_checks/    # schema and data quality checks
│           ├── io_managers/     # Parquet IO manager
│           ├── preprocessing/   # pure transformation functions
│           ├── resources/       # data loader, MLflow, training config
│           ├── training/        # model factory, metrics, splitting
│           └── utils/           # metadata and git helpers
├── subjects/
├── tests/
├── Makefile
└── pyproject.toml
```

## Main Technologies

* Python 3.11
* pandas
* scikit-learn
* XGBoost
* LightGBM
* Dagster
* MLflow
* Jupyter Notebooks
* Ruff
* Pytest
* uv

## Main Topics

* exploratory data analysis
* data preprocessing and validation
* feature engineering
* workflow orchestration
* config-driven ML pipelines
* experiment tracking and model registry
* forecasting systems
* reproducibility
* MLOps foundations

## Reports

Detailed reports and observations can be found in the `reports/` directory.