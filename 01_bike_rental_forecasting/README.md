# Bike Rental Forecasting

End-to-end machine learning engineering project focused on preparing and forecasting city-wide bike rental demand.

This project is part of the appliedAI ML & MLOps track and evolves over multiple weeks from exploratory data analysis and preprocessing to model training, workflow orchestration, and MLOps workflows.

## Quick Start

```bash
uv sync --dev
uv run dg dev
```

## Project Goals

The project aims to:

* prepare operational bike rental data for machine learning
* build reproducible preprocessing pipelines
* engineer useful temporal and contextual features
* train forecasting models for bike rental demand
* integrate ML workflows into structured pipelines
* apply MLOps practices such as testing, experiment tracking, and deployment

## Current Scope

### Week 2 — Data Pipeline ✅

Implemented components include:

* source dataset ingestion through reusable Dagster resources
* custom CSV IO manager for asset persistence
* source dataset schema validation with asset checks
* hourly rental activity aggregation
* weather anomaly correction and humidity imputation
* calendar feature generation
* unified bike rental feature dataset creation
* structured metadata tracking and logging
* reusable preprocessing transformations
* automated tests and CI validation

### Week 3 — EDA and Bike Rental Predictions ✅

Implemented components include:

* comprehensive exploratory data analysis
* data quality validation and consistency checks
* baseline forecasting models
* cyclical temporal feature engineering
* historical demand lag features
* contextual historical demand aggregations
* feature selection and evaluation
* Random Forest forecasting model
* XGBoost forecasting model
* feature importance analysis
* model comparison and benchmarking
* Dagster-based feature engineering assets
* chronological train-test splitting assets
* automated model training assets
* automated model evaluation assets

Results:

* XGBoost achieved an R² score above 0.918
* historical demand features provided the strongest predictive signal
* the complete forecasting workflow was integrated into Dagster

## Planned Future Work

* experiment tracking with MLflow
* model versioning and management
* workflow orchestration improvements
* inference and prediction workflows
* monitoring and model performance tracking
* deployment-oriented MLOps workflows

## Project Structure

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── quarantine/
├── notebooks/
├── reports/
├── src/
├── subjects/
└── tests/
```

## Main Technologies

* Python
* pandas
* scikit-learn
* XGBoost
* Dagster
* MLflow
* Jupyter Notebooks
* Ruff
* Pytest
* uv

## Main Topics

* exploratory data analysis
* data preprocessing
* data validation
* quarantine handling
* feature engineering
* workflow orchestration
* machine learning pipelines
* forecasting systems
* reproducibility
* MLOps foundations

## Reports

Detailed reports and observations can be found in the `reports/` directory.
