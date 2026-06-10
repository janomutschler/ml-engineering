# Bike Rental Forecasting

End-to-end machine learning engineering project focused on preparing and forecasting city-wide bike rental demand.

This project is part of the appliedAI ML & MLOps track and evolves over multiple weeks from exploratory data analysis and preprocessing to model training, workflow orchestration, and MLOps workflows.

## Problem

A city-wide bike sharing company needs to plan how many bikes to allocate across the city each day. The bike distribution department makes this decision one day in advance, so they need a **day-ahead forecast of hourly rental demand** for the upcoming day.

This project builds that forecast as a reproducible, production-style ML system: it ingests historical rental, weather, and holiday data, engineers temporal and demand-history features, trains and evaluates forecasting models, and serves a **next-day hourly demand prediction** through an HTTP API. The scope is deliberately the next day only — the prediction horizon is the 24 hours immediately following the most recent observed data, which is what the distribution department's planning cycle requires.

## Quick Start

### Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- make
- git
- Docker (for LakeFS)

**macOS only** — XGBoost and LightGBM require `libomp` from Homebrew:
```bash
brew install libomp
```

### Install

```bash
make install
```

### Configure

Copy the environment template and fill in your values:

```bash
cp .env.example .env
```

The defaults in `.env.example` match the local dev setup out of the box — no changes needed unless you're pointing at a remote server.

### Run

The project requires three services. Start them in order:

**Step 1 — LakeFS (data versioning):**

```bash
make infra        # starts LakeFS in Docker (background)
make lakefs-repo  # create the repository (run once)
```

LakeFS UI available at `http://localhost:8000`.

**Step 2 — MLflow tracking server (separate terminal):**

```bash
make mlflow
```

MLflow UI available at `http://127.0.0.1:5000`.

**Step 3 — Dagster development server (separate terminal):**

```bash
make dev
```

Dagster UI available at `http://127.0.0.1:3000`. Materialize assets from the asset graph to run the pipeline end to end.

**Step 4 — Prediction API (separate terminal, after a model is promoted):**

```bash
make api
```

The forecast API is available at `http://127.0.0.1:8001`, with interactive docs at `http://127.0.0.1:8001/docs`. It loads the `@champion` model from the MLflow registry, so the pipeline must have run and promoted a model at least once. Example request:

```bash
uv run python scripts/example_request.py
```

### Other commands

```bash
make infra-down   # stop LakeFS
make test         # run the test suite
make lint         # check code style and formatting
make format       # auto-fix formatting and linting
```

## Project Goals

The project aims to:

* prepare operational bike rental data for machine learning
* build reproducible preprocessing pipelines
* engineer useful temporal and contextual features
* train forecasting models for bike rental demand
* integrate ML workflows into structured pipelines
* apply MLOps practices such as experiment tracking, model versioning, data versioning, and deployment

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

* every training run logged to MLflow with params, metrics (MAE, RMSE, RMSLE, R²), model signature, input example, and provenance tags (Dagster run ID, git commit)
* trained model registered as a versioned artifact in the MLflow model registry (`bike_rental_forecaster`)
* `MlflowResource` owns the connection and run lifecycle; assets decide what to log
* tracking URI injected via `MLFLOW_TRACKING_URI` environment variable (fail-loud, no silent fallback)
* `mlflow.sklearn.log_model` used uniformly across all model types via the sklearn-compatible API

### Phase 2 — Walk-forward Backtesting ✅

* single holdout evaluation replaced with 5-fold expanding-window walk-forward cross-validation via `TimeSeriesSplit`
* backtest and final training unified in one MLflow run: cross-validation first (honest performance estimate), then fit on all data (the registered artifact)
* per-fold metrics logged step-indexed to MLflow (trajectory chart); `mean_*` and `std_*` aggregates as run-level summary
* `std_r2` exposes model stability across time — not possible with a single holdout
* fold 1 cold-start behaviour documented: steady-state performance (folds 2–5) is ~0.90 ± 0.025; full mean including fold 1 is ~0.852

### Phase 3 — Champion/Challenger Promotion Gate ✅

* `model_promotion` asset compares the newly registered version against the current `@champion` on `mean_r2`
* first run bootstraps the champion automatically; subsequent runs promote only on strict improvement
* `@challenger` alias assigned to versions that don't beat the champion
* promotion policy (`metric`, `higher_is_better`, `min_improvement`) is a per-run `PromotionConfig` overridable from the launchpad
* production and inference always load `models:/bike_rental_forecaster@champion` — no hardcoded version numbers

### Phase 4 — LakeFS Data Versioning and Full Lineage ✅

* LakeFS runs locally via Docker, backed by named volumes; production swap to S3/GCS requires only a storage namespace config change
* `LakeFSParquetIOManager` writes all DataFrame assets to a per-run branch (`dagster-<run_id>`), created as a zero-copy snapshot of `main`
* `data_version` asset commits the run branch and merges to `main` after all data assets materialise — nothing reaches `main` unless the full run completes
* partial re-runs (e.g. retrain only) read the last published snapshot from `main` via a fresh branch, with no upstream re-runs required
* every MLflow training run is tagged with `lakefs_commit`, completing the lineage chain: registered model → MLflow run → git commit + LakeFS data commit → exact Parquet snapshot

### Phase 5 — Prediction API ✅

* FastAPI service exposes a day-ahead demand forecast: 24 hourly predictions for the period immediately following the last published data point
* the `@champion` model is loaded from the MLflow registry by alias — deploying a new model is an alias move plus a `/reload`, never a code change or redeploy
* features for the forecast horizon are computed server-side by reusing the *exact* pipeline transform functions, eliminating training-serving skew; backward-looking lag features are derived from history published to LakeFS `main`
* request validation via Pydantic (24 hours required, known conditions only, range-checked weather); feature columns read from the model's logged signature
* `/health` and every forecast response carry the served `model_version` and `data_commit`, surfacing the full lineage chain at the serving layer

## Planned Future Work

* file-arrival sensor for automated retraining on new data
* input drift and performance decay monitoring
* deployment-oriented MLOps workflows

## Project Structure

```text
.
├── data/
│   ├── sources/         # raw input CSVs (not tracked by git)
│   ├── processed/       # local fallback for non-LakeFS assets
│   └── quarantine/
├── mlartifacts/         # MLflow artifact store
├── mlflow.db            # MLflow tracking database (SQLite)
├── notebooks/
├── reports/
├── scripts/
│   └── bootstrap_lakefs.py   # idempotent repo creation
├── src/
│   └── bike_rental/
│       ├── api/             # FastAPI prediction service
│       └── defs/
│           ├── assets/          # Dagster asset definitions
│           ├── asset_checks/    # schema and data quality checks
│           ├── io_managers/     # LakeFS-backed Parquet IO manager
│           ├── preprocessing/   # pure transformation functions
│           ├── resources/       # data loader, MLflow, LakeFS, training config
│           ├── training/        # model factory, metrics, backtesting
│           └── utils/           # metadata and git helpers
├── subjects/
├── tests/
├── docker-compose.yml
├── Makefile
├── .env.example
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
* LakeFS
* FastAPI
* Docker
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
* walk-forward backtesting
* champion/challenger model promotion
* data versioning and branching
* end-to-end ML lineage
* model serving and inference APIs
* reproducibility
* MLOps foundations

## Reports

Detailed reports and observations can be found in the `reports/` directory.