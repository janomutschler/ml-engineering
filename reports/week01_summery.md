# Week 1 - EDA Summary

## Objective
Explore the Titanic dataset and identify relationships between passenger features and survival outcomes.

## Key Findings
- Sex shows the strongest correlation with survival.
- First-class passengers survived more frequently than third-class passengers.
- Higher fare values correlate with higher survival probability.
- Age alone shows only weak linear correlation with survival, although younger passengers exhibit higher survival rates.

## Preprocessing Decisions
- Apply feature scaling to numerical features (`age`, `fare`, `family_size`).
- Retain one-hot encoded class features.
- Split dataset into training and testing subsets for model development.

## Next Steps
- Build a logistic regression baseline using scikit-learn.
- Implement logistic regression from scratch using NumPy.