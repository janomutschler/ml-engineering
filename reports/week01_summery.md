# Week 1 - Classification Summary

## Objective
Explore the Titanic dataset and build a baseline binary classification model to predict passenger survival using logistic regression.

## Exploratory Data Analysis Findings
- Sex shows the strongest correlation with survival.
- First-class passengers survived more frequently than third-class passengers.
- Higher fare values correlate with higher survival probability.
- Age alone shows only weak linear correlation with survival, although younger passengers exhibit higher survival rates.

## Preprocessing Decisions
- Split the dataset into feature variables (`X`) and target labels (`y`).
- Divide the data into training and test sets using stratified sampling.
- Apply feature scaling to numerical features (`age`, `fare`, `family_size`) using `StandardScaler`.
- Retain binary encoded categorical features without scaling.

## Logistic Regression Baseline
A logistic regression baseline was implemented using scikit-learn.

### Evaluation Results
- Accuracy: 0.787
- Precision: 0.754
- Recall: 0.667
- F1 Score: 0.708

The model achieved solid baseline performance using a relatively small feature set and establishes a reference point for the later NumPy-based implementation.

## Next Steps
- Implement logistic regression from scratch using NumPy.
- Compare custom implementation results with the scikit-learn baseline.
- Add additional evaluation and visualization improvements.