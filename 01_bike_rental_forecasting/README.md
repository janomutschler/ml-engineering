# Bike Rental Forecasting

End-to-end machine learning engineering project focused on preparing and later
forecasting city-wide bike rental demand.

This project is part of the appliedAI ML & MLOps track and evolves over
multiple weeks from exploratory data analysis and preprocessing to model
training, workflow orchestration, and MLOps workflows.

## Project Goals

The project aims to:

- prepare operational bike rental data for machine learning
- build reproducible preprocessing pipelines
- engineer useful temporal and contextual features
- train forecasting models for bike rental demand
- integrate ML workflows into structured pipelines
- apply MLOps practices such as testing, experiment tracking, and deployment

## Current Scope

### Week 2 — Data Pipeline

The current implementation focuses on designing and implementing a reusable
preprocessing pipeline for downstream forecasting workflows.

Implemented components include:

- exploratory preprocessing workflow design in notebooks
- modular Dagster asset pipeline implementation
- raw operational, weather, and holiday dataset ingestion
- structured dataset validation and quarantine handling
- hourly aggregation of rental activity
- weather and holiday enrichment
- temporal feature engineering
- curated base dataset materialization
- custom Dagster CSV IO manager integration
- structured metadata tracking and logging
- automated tests and CI validation

### Planned Future Work

- forecasting model training
- time-series evaluation workflows
- experiment tracking with MLflow
- workflow orchestration improvements
- feature store and model management concepts
- inference and deployment workflows

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

- Python
- pandas
- scikit-learn
- Dagster
- MLflow
- Jupyter Notebooks
- Ruff
- Pytest
- uv

## Main Topics

- data preprocessing
- data validation
- quarantine handling
- feature engineering
- workflow orchestration
- machine learning pipelines
- forecasting systems
- reproducibility
- MLOps foundations

## Reports

Detailed reports and observations can be found in the `reports/` directory.