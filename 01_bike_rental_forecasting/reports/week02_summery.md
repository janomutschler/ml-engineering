# Week 2 - Data Pipeline

## Summary

In this assignment, a reusable preprocessing pipeline for a bike rental
forecasting project was designed and implemented using Dagster.

The workflow processes operational rental, weather, and holiday datasets and
transforms them into a curated base dataset for downstream analysis and machine
learning workflows.

The assignment initially focused on exploring and prototyping the preprocessing
workflow inside notebooks before implementing the final reusable Dagster
pipeline architecture.

The implementation focused on building modular preprocessing assets, structured
data validation, quarantine handling for invalid records, and reproducible
dataset materialization.

---

## Pipeline Architecture

The Dagster pipeline consists of multiple modular preprocessing stages:

- loading raw source datasets
- validating required schema and timestamps
- quarantining invalid records
- aggregating rental events into hourly demand
- enriching rental demand with weather information
- enriching rental demand with holiday information
- deriving calendar-based ML features
- materializing the curated base dataset

The pipeline was implemented using reusable transformation functions and
Dagster assets with metadata tracking and structured logging.

---

## Validation and Quarantine Handling

The preprocessing workflow includes validation checks for:

- missing required columns
- invalid timestamps
- duplicate identifiers
- duplicate weather timestamps
- invalid location identifiers
- duplicate holiday dates

Invalid records are separated into quarantine datasets together with
human-readable quarantine reasons for later inspection and debugging.

---

## Final Dataset Design

The resulting base dataset represents:

- one location
- during one hourly time window

The dataset combines:

- hourly rental demand
- temporal features
- weather information
- holiday indicators

Derived temporal features include:

- `hour`
- `weekday`
- `month`
- `is_weekend`

The primary target variable for later forecasting workflows is:

- `total_rentals`

---

## Engineering Improvements

The implementation additionally introduced:

- Dagster asset lineage
- custom CSV IO manager integration
- reusable metadata helpers
- automated dataset materialization
- unit tests for preprocessing and aggregation logic
- CI validation using Ruff and Pytest

---

## Next Steps

The next step is to build forecasting-oriented machine learning workflows on top
of the curated base dataset, including model training, evaluation, and workflow
orchestration.