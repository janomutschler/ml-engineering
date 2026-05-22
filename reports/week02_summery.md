# Week 2 - Data Pipeline

## Summary

In this assignment, a preprocessing workflow for a bike rental forecasting
project was explored and prototyped using operational, weather, and holiday
datasets.

The work focused on validating the available data sources, aggregating rental
events into hourly activity per location, and designing the structure of the
final ML-ready dataset.

---

## Workflow Overview

The preprocessing workflow included:

- loading and validating the source datasets
- converting and validating datetime information
- aggregating rental events into hourly demand
- deriving temporal features
- enriching the dataset with weather and holiday information
- validating joins and temporal consistency

---

## Final Dataset Design

The final prototype dataset represents:

- one location
- during one hourly time window

The dataset combines:
- rental activity
- temporal features
- weather information
- holiday indicators

The primary target for later machine learning workflows is:

- `total_rentals`

---

## Key Observations

- The operational datasets could be successfully aggregated into hourly rental
  activity.
- Weather and holiday information could be joined successfully to the
  aggregated rental data.
- The resulting dataset provides a structured foundation for later forecasting
  and machine learning workflows.

---

## Next Steps

The next step is to implement the explored preprocessing workflow as a reusable
Dagster pipeline with structured assets and automated dataset materialization.