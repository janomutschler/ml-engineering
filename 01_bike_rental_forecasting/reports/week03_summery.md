# Week 3 Rental Predictions

## Summery

Week 3 focuses on extending the bike rental project from data preparation to machine learning and forecasting.

The assignment consists of four major stages:

Exploratory Data Analysis
Baseline Regression Modeling
Model Improvement and Feature Engineering
Pipeline Integration

At the current stage, the exploratory data analysis has been completed. The analysis validated the prepared dataset, investigated data quality, explored rental demand patterns, and identified promising predictor variables for future regression models.

The findings from this phase will guide the next steps of Week 3, including feature engineering, baseline model development, model evaluation, and integration of the final training workflow into the existing Dagster pipeline.

---

## Dataset Validation and Data Quality

The final dataset was validated to ensure consistency after integrating rental, weather, temporal, and holiday information.

Several data quality checks were performed, including:

* Duplicate record validation
* Missing value analysis
* Invalid value detection
* Temporal continuity checks
* Feature consistency verification

The analysis identified a small number of missing hourly periods caused by unavailable operational and weather observations. Additionally, isolated weather quality issues were detected, including invalid humidity values and unusual perceived temperature observations.

Overall, the dataset was found to be largely complete and suitable for further modeling work.

---

## Target Variable Analysis

The rental demand target variable (`total_rentals`) was analyzed to better understand its distribution and behavior.

Key observations included:

* A right-skewed demand distribution
* Frequent low-demand and zero-demand periods
* Strong variability between low and high rental hours

Long-term demand trends also revealed increasing rental activity over the two-year observation period together with recurring seasonal patterns.

---

## Temporal Demand Patterns

Temporal analysis revealed several important demand dynamics:

* Strong daily demand cycles
* Pronounced morning and evening rush-hour peaks
* Consistent weekday demand patterns
* Reduced rental activity on holidays
* Clear seasonal fluctuations throughout the year

These findings indicate that temporal features will likely be important predictors for future forecasting models.

---

## Weather Impact Analysis

The relationship between weather and rental demand was investigated using correlation analysis and aggregated demand trends.

Key findings included:

* Temperature shows a moderate positive relationship with rental demand.
* Rental activity generally increases as temperatures rise.
* Rental demand decreases as weather conditions worsen.
* Higher humidity levels are associated with lower rental activity.
* Windspeed exhibits only a relatively weak relationship with demand.

The analysis suggests that weather-related features provide valuable predictive information, with temperature appearing to be the strongest weather predictor.

---

## Feature Relationships and Modeling Considerations

Correlation analysis was used to identify important feature relationships and potential modeling risks.

Several observations emerged:

* Temperature and perceived temperature are almost perfectly correlated and may provide redundant information.
* Operational variables (`booked_rentals` and `direct_pickups`) are direct components of the target variable and would introduce target leakage if used as predictors.
* Temporal and weather-related features show meaningful relationships with rental demand and represent promising candidate predictors.

---

## Next Steps

The findings from the exploratory analysis establish a strong foundation for the next stage of the project.

Future work will focus on:

* Feature engineering
* Cyclical time encodings
* Lag and rolling-window features
* Train-test splitting strategy
* Baseline regression model development
* Forecasting-oriented evaluation methods

The insights gained during Week 3 will guide feature selection and model design throughout the remainder of the bike rental forecasting project.
