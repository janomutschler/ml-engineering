# Week 3 — Exploratory Analysis and Forecasting Models

> **Milestone:** understand the demand data, build a baseline, then iterate with feature engineering and stronger models — and fold the winning recipe back into the pipeline.

## Objective

With a clean hourly dataset in hand, this week shifts from data preparation to *learning from the data*. The aim is to predict a continuous target — hourly rental demand — which calls for regression models and time-aware evaluation. The work runs in three movements: explore, model, and integrate.

## Exploratory data analysis

The EDA focused on what drives demand and which predictors look promising. Key patterns:

- **Strong daily cycles** with pronounced morning/evening rush-hour peaks.
- **Stable weekly patterns** — weekdays and weekends behave distinctly and consistently.
- **Seasonal and long-term trends** across months.
- **Reduced demand on holidays.**
- **Weather relationships** — demand responds to conditions, temperature, and rain.

The headline takeaway: demand is *highly structured in time*, which strongly suggested that **historical demand itself** would be the most valuable predictor.

## Baseline

Two references were established before any tuning:

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Dummy Regressor (predicts the mean) | 174.98 | 232.61 | −0.11 |
| Linear Regression (raw features) | 134.30 | 199.58 | 0.18 |

Linear Regression beating the naive baseline confirmed the weather and calendar features carried real signal — a starting point to improve on, not a finish line.

## Feature engineering experiments

A series of time-boxed experiments, each justified by its effect:

- **Cyclical time encodings.** Hour and month encoded as sine/cosine pairs so the model sees that hour 23 is adjacent to hour 0. R² rose from **0.18 → 0.26**.
- **Historical-demand features** (the big lever):
  - Daily and weekly lags (`lag_24h`, `lag_168h`)
  - Same-hour rolling average over recent days
  - Same-weekday-hour rolling average over recent weeks

  Context-aware aggregations clearly beat generic rolling averages. The single strongest feature was the **same-weekday-hour historical average** — it captures "what demand usually looks like at this hour on this kind of day."

With the selected feature set, Linear Regression jumped to **R² ≈ 0.86**, evidence that most of the gain came from features rather than model complexity.

## Model comparison

Three models on the selected feature set, chronological holdout:

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | 50.74 | 82.74 | 0.86 |
| Random Forest | 44.92 | 72.28 | 0.89 |
| **XGBoost** | **40.17** | **62.88** | **0.92** |

**XGBoost** won, explaining ~92% of demand variability. Feature-importance analysis on the tree models confirmed the EDA intuition: historical-demand features dominate, weekly and daily lags are highly influential, and weather is useful but secondary.

## Leakage-awareness

Because the strongest features are historical, care was taken to keep them **causal**: every lag and rolling aggregate looks strictly backward (shifted so the current row never sees its own target). This is what later made it safe to compute features once over the full series and evaluate by time-ordered splits — a row at time *t* only depends on data up to *t*.

## Pipeline integration

The winning recipe was integrated into the Dagster pipeline so feature generation, training, and evaluation became reproducible assets rather than notebook cells — extending the Week 2 graph with a modeling stage that materializes end to end.

## What I learned

The value of establishing a baseline before optimizing; how to run disciplined, hypothesis-driven feature-engineering experiments and quantify each one; why time-series problems need causal features and time-aware validation; and how to read feature importance to confirm (or challenge) what the EDA suggested. The biggest lesson: **feature engineering outperformed model selection** here.

## How this evolved later

The Week 3 integration used a single chronological train/test split and a separate evaluation step. In the MLOps phase this was replaced by **walk-forward backtesting** (expanding-window cross-validation), which reports performance as a distribution over time and exposes stability (`std_r2`) that a single split cannot. Model selection also became **config-driven** — XGBoost, LightGBM, Random Forest, Linear Regression, and Dummy behind one typed configuration — but the selected feature set and the conclusion that XGBoost is the strongest candidate carried forward.