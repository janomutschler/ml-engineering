# Bike Rental Forecasting

End-to-end machine learning engineering project focused on preparing and later forecasting city-wide bike rental demand.

This project is part of the appliedAI ML & MLOps track and evolves over multiple weeks from data preprocessing to model training and MLOps workflows.

## Project Goals

The project aims to:
- prepare operational bike rental data for machine learning
- build reproducible preprocessing pipelines
- engineer useful temporal and contextual features
- train forecasting models for bike rental demand
- integrate ML workflows into structured pipelines
- apply MLOps practices such as experiment tracking and deployment

## Current Scope

### Week 2 — Data Pipeline
- exploratory data analysis on rental and weather data
- preprocessing workflow design
- hourly aggregation of rental activity
- feature engineering
- weather and holiday enrichment
- Dagster asset pipeline implementation
- CSV dataset materialization

### Planned Future Work
- machine learning model training
- forecasting evaluation
- experiment tracking with MLflow
- workflow orchestration improvements
- inference and deployment workflows

## Structure

```text
.
├── data/
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
- uv

## Main Topics

- data preprocessing
- feature engineering
- workflow orchestration
- machine learning pipelines
- forecasting systems
- reproducibility
- MLOps foundations

## Reports

Detailed reports and observations can be found in the `reports/` directory.