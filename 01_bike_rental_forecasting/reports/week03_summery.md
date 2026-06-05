# Week 3 Rental Predictions

## Summary

Week 3 extends the bike rental project from data preparation into machine learning and demand forecasting.

The assignment focused on four major stages:

* Exploratory Data Analysis
* Baseline Regression Modeling
* Feature Engineering and Model Development
* Pipeline Integration

During this phase, the prepared dataset was validated, explored, and used to develop and evaluate multiple forecasting models. A structured experimentation workflow was implemented to compare baseline models, feature engineering techniques, and machine learning algorithms.

The work resulted in a high-performing forecasting model capable of explaining more than 90% of the observed variability in bike rental demand. In addition, the experiments provided valuable insights into the factors that drive rental activity and the relative importance of different feature engineering approaches.

---

## Dataset Validation and Data Quality

The final dataset was validated after integrating rental, weather, temporal, and holiday information through the preprocessing pipeline.

Several data quality checks were performed, including:

* Duplicate record validation
* Missing value analysis
* Suspicious value detection
* Temporal continuity checks
* Feature consistency verification

The analysis confirmed that the dataset contains no duplicate records or missing values. A small number of missing hourly timestamps were identified due to periods without available source data, but these gaps represent only a minor fraction of the overall observation period.

Additional validation confirmed that rental counts, temporal features, and categorical variables were internally consistent and suitable for forecasting applications.

---

## Exploratory Data Analysis

The exploratory analysis focused on understanding demand behavior and identifying promising predictor variables.

Key findings included:

* Strong daily demand cycles with pronounced rush-hour peaks
* Consistent weekly demand patterns
* Long-term growth trends throughout the observation period
* Seasonal fluctuations across different months
* Reduced demand during holidays
* Meaningful relationships between weather conditions and rental activity

The analysis revealed that bike rental demand exhibits highly structured temporal behavior, suggesting that historical demand information would likely play an important role in future forecasting models.

---

## Baseline Modeling

Two baseline models were evaluated:

* Dummy Regressor
* Linear Regression

The Dummy Regressor predicts the average rental demand for all observations and serves as a naive benchmark.

| Model                        | MAE    | RMSE   | R²    |
| ---------------------------- | ------ | ------ | ----- |
| Dummy Regressor              | 174.98 | 232.61 | -0.11 |
| Linear Regression (Baseline) | 134.30 | 199.58 | 0.18  |

The Linear Regression model substantially outperformed the naive baseline, demonstrating that the available weather and temporal features contain meaningful predictive information.

---

## Feature Engineering Experiments

A series of feature engineering experiments were conducted to improve forecasting performance.

### Cyclical Temporal Features

Hour and month variables were transformed using sine and cosine encodings to better represent cyclical temporal patterns.

This increased the model R² score from 0.18 to 0.26.

### Historical Demand Features

Historical demand features produced the largest performance gains.

Implemented features included:

* Daily lag (`lag_24h`)
* Weekly lag (`lag_168h`)
* Generic rolling averages
* Same-hour historical averages
* Same weekday-hour historical averages

The experiments demonstrated that context-aware historical demand features significantly outperform generic rolling statistics.

The strongest individual feature was the same weekday-hour historical average, which captures recurring demand behavior observed during similar temporal conditions.

### Feature Selection

Based on the experiment results, a final feature set was selected consisting of:

* Weather features
* Cyclical temporal features
* Daily and weekly lag features
* Same-hour historical averages
* Same weekday-hour historical averages

Generic rolling averages were excluded because they provided only marginal performance improvements compared to the more targeted historical aggregation features.

Using the selected feature set, Linear Regression achieved an R² score of approximately 0.86.

---

## Model Evaluation

Three forecasting algorithms were evaluated using the selected feature set:

| Model                                 | MAE   | RMSE  | R²   |
| ------------------------------------- | ----- | ----- | ---- |
| Linear Regression (Selected Features) | 50.74 | 82.74 | 0.86 |
| Random Forest                         | 44.92 | 72.28 | 0.89 |
| XGBoost                               | 40.17 | 62.88 | 0.92 |

XGBoost achieved the strongest overall performance, explaining more than 91% of the observed variability in bike rental demand.

The comparison also showed that feature engineering contributed substantially more to model performance than model complexity alone.

---

## Feature Importance Analysis

Feature importance analysis was performed using the tree-based models.

The results showed that historical demand features dominate the predictive signal.

In particular:

* Same weekday-hour historical averages were the most important predictor
* Weekly lag features were highly influential
* Daily lag features contributed substantial predictive value
* Weather variables provided useful but secondary information

These findings confirm the conclusions from the exploratory analysis and feature engineering experiments.

---

## Pipeline Integration

The forecasting workflow was integrated into the existing Dagster pipeline to enable reproducible feature generation, model training, and model evaluation.

Implemented assets include:

* Modeling feature set generation
* Cyclical temporal feature creation
* Historical demand feature generation
* Chronological train-test splitting
* XGBoost model training
* Automated model evaluation

The resulting asset graph extends the Week 2 preprocessing pipeline with a complete machine learning workflow, allowing the forecasting process to be materialized end-to-end through Dagster.

---

## Key Findings

Several important conclusions emerged from the forecasting workflow:

* Historical demand is the strongest predictor of future rental activity.
* Context-aware historical aggregations outperform generic rolling statistics.
* Weekly and daily demand patterns are highly stable and predictable.
* Weather contributes useful information but is less influential than temporal demand behavior.
* Feature engineering provides larger performance gains than model selection alone.
* XGBoost achieved the strongest forecasting performance among all evaluated models.
* The final forecasting workflow was successfully integrated into Dagster and can be executed as a reproducible asset pipeline.

---

## Next Steps

The results of Week 3 establish a strong forecasting foundation for the remainder of the project.

Future work will focus on:

* Experiment tracking with MLflow
* Model versioning and management
* Automated retraining workflows
* Workflow orchestration improvements
* Inference and prediction workflows
* Monitoring and maintaining forecasting performance over time
* Deployment-oriented MLOps workflows

The selected feature set and XGBoost model provide a strong candidate for future production-oriented forecasting and MLOps workflows.
