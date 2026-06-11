# Week 4 — Reproducible Pipelines and the MLOps System

> **Milestone:** turn a good model into a production-style system — tracked experiments, a versioned model registry with automated promotion, versioned data with full lineage, and a served prediction API.

## Objective

Training a model is only part of the job. A real ML system must answer operational questions: *Which data produced this model? Which version is in production? How do we ship a better one without a redeploy? How do we serve predictions reliably?* This week (and the production-readiness work that followed it) wires up the MLOps backbone that answers all of them, and refactors the training code to production quality.

## 1. Experiment tracking with MLflow

Every training run is logged to MLflow as the system of record:

- **Parameters** — model type, hyperparameters, fold count, feature count, target.
- **Metrics** — MAE, RMSE, RMSLE, and R², logged **per backtest fold** (step-indexed, so the UI draws a trajectory) plus run-level `mean_*`/`std_*` aggregates.
- **Artifacts** — the fitted model with an inferred **signature** (input columns/types) and an **input example**.
- **Provenance tags** — the Dagster run ID, git commit, and LakeFS data commit.

A dedicated `MlflowResource` owns the connection and run lifecycle (it opens/closes the run and marks it FAILED on error); assets decide *what* to log. The tracking URI is injected via an environment variable that fails loudly if unset.

## 2. Model registry, versioning, and promotion

Each run registers the model as a new version of `bike_rental_forecaster` in the MLflow registry. A **champion/challenger promotion gate** decides what gets served:

- The newly registered version is compared against the current `@champion` on `mean_r2`.
- The first run **bootstraps** the champion automatically.
- A later version is promoted to `@champion` only if it **strictly improves** on the incumbent (beyond a configurable margin); otherwise it's recorded as `@challenger` and the champion is untouched.
- The policy (metric, direction, minimum improvement) is configurable per run.

Production and inference always load `models:/bike_rental_forecaster@champion` — there are no hardcoded version numbers, so **deploying a new model is an alias move**, not a code change. The promotion logic is split into a pure decision function and a thin registry-orchestration layer, both unit-tested.

## 3. Honest evaluation: walk-forward backtesting

The single chronological holdout from Week 3 was replaced with **expanding-window walk-forward cross-validation**. Each fold trains on history and scores the next window, so performance is a *distribution* across time. The same MLflow run records:

1. the per-fold metrics (the honest generalization estimate), then
2. the final model fit on **all** data — the registered artifact.

The deployed model uses the full history; the backtest, not a held-out slice, is its performance estimate. `std_r2` surfaces stability over time, and the fold-1 cold start is documented rather than hidden (steady-state ≈ 0.90, full mean ≈ 0.85).

## 4. Data versioning with LakeFS

Data is versioned with LakeFS using a deliberate **branch-as-transaction** strategy:

- The Parquet IO manager writes every data asset to a per-run branch, `dagster-<run_id>`, created as a **zero-copy snapshot** of `main`.
- A `data_version` asset **commits** that branch and **merges** it to `main` only after all data assets have materialized — so `main` only ever advances to a complete, consistent dataset (no half-updated state).
- No-op runs are handled gracefully (an unchanged branch returns the current head instead of failing).
- Partial re-runs (e.g. *retrain only*) read the last published snapshot from `main` via a fresh branch, with no upstream re-runs required.

This answers the assignment's two key questions directly: *which dataset version trained this model?* (the `lakefs_commit` tag on the MLflow run) and *how is `main` protected?* (nothing reaches it until a run's data assets complete, and the merge is the single publish gate).

## 5. End-to-end lineage

The above combine into one traceable chain:

```
registered model version
   └─ MLflow run (params, metrics, signature)
        ├─ git commit     (the code)
        └─ LakeFS commit  (the exact Parquet snapshot of every data asset)
```

Any model — and, through the API, any prediction — can be traced back to the exact code and data that produced it.

## 6. Prediction API

A FastAPI service exposes the day-ahead forecast: 24 hourly predictions for the period immediately following the last published data point.

- **Dynamic model loading.** The `@champion` model is loaded from the registry by alias at startup; a `/reload` endpoint picks up a newly promoted champion without a restart.
- **No training–serving skew.** Horizon features are computed server-side by reusing the *exact same* feature-assembly functions used in training; backward-looking lag features are derived from history published to LakeFS `main`. A parity test pins the two paths to identical output.
- **Validated inputs.** Pydantic enforces exactly 24 hourly entries, known weather conditions only, and range-checked values; the model's feature columns are read from its logged signature rather than hardcoded.
- **Lineage at the edge.** `/health` and every forecast response carry the served `model_version` and `data_commit`.

## 7. Production-readiness refactors

Alongside the MLOps wiring, the codebase was hardened:

- **Config-driven model selection** via a typed `TrainingConfigResource` — every model and hyperparameter is discoverable in the Dagster launchpad.
- **Pure core extracted** — the backtest-and-fit logic lives in a framework-free function, so the heart of the project is unit-testable without standing up MLflow.
- **Shared feature assembly** — one definition of the feature pipeline, called by both training and serving.
- **A focused test suite** — transforms, feature parity, metrics aggregation, the model factory, API schemas/features, the promotion gate, and the training core.

## What I learned

The shape of a real MLOps system: experiment tracking and a model registry as the source of truth; alias-based promotion and deployment that decouples shipping a model from shipping code; data versioning with a branching strategy that makes publishes atomic and reads reproducible; and an end-to-end lineage chain that makes the whole system auditable. On the engineering side: serving a model without training–serving skew, designing for testability by separating pure logic from framework glue, and surfacing configuration so the system is operable, not opaque.

## Future work

- A file-arrival sensor to retrain automatically when new source data lands (closing the loop to a hands-off pipeline).
- Input-drift and performance-decay monitoring.
- Deployment-oriented packaging against remote LakeFS/MLflow and object storage.