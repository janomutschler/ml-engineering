# Week 2 — Data Pipeline

> **Milestone:** turn four raw operational CSVs into a clean, reproducible, hourly feature dataset, built as a Dagster pipeline with proper separation of concerns.

## Objective

The bike-sharing company plans bike allocation a day in advance and wants a demand forecast to support it. Before any model can be trained, the raw data has to be ingested, cleaned, aligned to a common hourly grid, and enriched with contextual features. This week delivers that foundation: a **reusable preprocessing pipeline** rather than a one-off script, so the dataset can be regenerated reliably whenever new data arrives.

## Inputs

Four sources extracted from the company's systems:

| Source | Contents |
|--------|----------|
| Booked rentals | Individual bikes reserved in advance |
| Direct pickups | Individual bikes taken without a booking |
| Weather | Hourly conditions, temperature, perceived temperature, humidity, wind |
| Holidays | Calendar of holidays over the project window |

## What was built

A Dagster asset graph that loads, validates, transforms, and persists the data:



The processing steps:

- **Aggregate to hourly activity.** Booked and direct rental *events* are floored to the hour and counted into `booked_rentals`, `direct_pickups`, and `total_rentals`. The timeline is filled to a continuous hourly grid so hours with zero demand are explicitly represented (not silently missing).
- **Clean the weather.** Implausible perceived-temperature readings (>20 °C from actual) are corrected, zero-humidity readings are treated as missing and interpolated, and timestamps are floored to the hour for joining.
- **Engineer calendar features.** From the hourly timeline: `hour`, `weekday`, `month`, `is_weekend`, and `is_holiday`.
- **Assemble the feature dataset.** Activity, cleaned weather, and calendar features are joined into one hourly table; the forecasting target is `total_rentals`.

## Engineering decisions

The week was as much about *structure* as about transforms — the goal was a pipeline that reads like a production system:

- **Separation of responsibilities.** Each Dagster concept was used for its purpose: **assets** for the data-processing steps, **resources** for configuration and external systems (the file-system data loader), and **IO managers** for persistence. Transformation logic lives in pure, importable functions; assets stay thin and declarative.
- **Schema validation as asset checks.** Each source has a check that its required columns are present, so a malformed input fails fast and visibly rather than corrupting downstream steps.
- **Continuity check.** An asset check verifies the hourly timeline has no missing hours.
- **Observability.** Every asset attaches metadata (row counts, dtypes, a preview, and step-specific counts like corrected/imputed values), and logs structured messages — so a run is inspectable in the Dagster UI.
- **Tests + CI hygiene.** Unit tests cover the aggregation, calendar, and weather transforms; Ruff enforces style and formatting.

## Outcome

A reproducible, observable pipeline that materializes end to end in Dagster and produces a clean hourly feature dataset with no duplicates or missing values (aside from a few genuinely-absent source hours). This dataset is the foundation everything later builds on.

## What I learned

How to model a data workflow as a dependency graph of assets; the discipline of keeping transformation logic separate from orchestration; using checks and metadata to make a pipeline trustworthy and observable; and handling real-world data-quality issues (gaps, anomalies, imputations) explicitly rather than implicitly.

## How this evolved later

The Week 2 design persisted intermediate datasets through a custom **CSV** IO manager. In the later MLOps work this was replaced by a **Parquet** IO manager backed by **LakeFS**, which removed datetime re-parsing and added full data versioning — but the asset structure, separation of concerns, and checks established here carried through unchanged.