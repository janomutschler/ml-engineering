# Week 2 - Data Pipeline

## Summary

In this assignment, a reusable preprocessing pipeline for a bike rental
forecasting project was designed and implemented using Dagster.

The workflow loads operational rental, weather, and holiday datasets and
transforms them into a curated feature dataset for downstream analysis and
machine learning workflows.

The assignment initially focused on exploring and prototyping preprocessing
steps in notebooks before implementing a production-oriented Dagster pipeline
using assets, resources, asset checks, metadata tracking, and structured
logging.

---

## Pipeline Architecture

The Dagster pipeline consists of multiple modular preprocessing stages:

* loading source datasets through a reusable data loader resource
* validating source dataset schemas with asset checks
* aggregating rental events into continuous hourly rental activity
* cleaning and validating weather observations
* generating calendar-based features
* combining rental activity, weather, and calendar information into a unified
  feature dataset
* persisting intermediate and final datasets through a custom CSV IO manager

The pipeline uses reusable transformation functions together with Dagster
assets, resources, metadata tracking, and structured logging.

The final asset graph consists of:

```text
booked_rentals
direct_pickups
weather
holidays
    ↓
hourly_rental_activity
weather_cleaned
calendar_features
    ↓
bike_rental_features
```

---

## Data Quality and Validation

The preprocessing workflow includes validation checks for:

* required source dataset columns
* continuous hourly rental activity

Additional preprocessing steps include:

* correction of unrealistic perceived temperature values
* imputation of invalid humidity measurements
* automatic datetime parsing during dataset loading
* metadata tracking for all intermediate datasets

The weather preprocessing workflow identifies anomalous records and applies
simple corrective actions to improve downstream data quality while preserving
the overall temporal structure of the dataset.

---

## Feature Engineering

Rental activity is aggregated into hourly demand observations and transformed
into a continuous hourly timeline to ensure that periods without rental activity
remain represented in the final dataset.

Calendar-based features are generated from the hourly timeline and include:

* `hour`
* `weekday`
* `month`
* `is_weekend`
* `is_holiday`

Weather observations are cleaned and joined to the hourly rental activity,
providing contextual information that may help future forecasting models
capture weather-related demand patterns.

---

## Final Dataset Design

The resulting feature dataset represents:

* one hourly time window

The dataset combines:

* hourly rental activity
* weather observations
* calendar-based features

Included weather features:

* `conditions`
* `temperature_c`
* `perceived_temperature_c`
* `humidity`
* `windspeed_kmh`

Included calendar features:

* `hour`
* `weekday`
* `month`
* `is_weekend`
* `is_holiday`

Operational rental metrics include:

* `booked_rentals`
* `direct_pickups`

The primary target variable for later forecasting workflows is:

* `total_rentals`

---

## Engineering Improvements

The implementation additionally introduced:

* Dagster asset lineage and dependency management
* reusable source data loader resource
* custom CSV IO manager integration
* structured asset metadata tracking
* Dagster asset checks
* reusable preprocessing transformation functions
* automated unit tests
* CI validation using Ruff and Pytest

The overall architecture was refactored following mentor feedback to improve
modularity, reproducibility, and production readiness.

---

## Next Steps

The next step is to build forecasting-oriented machine learning workflows on top
of the prepared feature dataset.

Future work will focus on:

* exploratory data analysis
* additional feature engineering
* forecasting model development
* model evaluation and comparison
* workflow orchestration for training and evaluation
* experiment tracking and MLOps workflows
