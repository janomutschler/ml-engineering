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

## Logistic Regression from Scratch

A complete logistic regression implementation was developed from scratch using NumPy without relying on scikit-learn model utilities.

The implementation includes:
- custom train/test splitting
- feature standardization
- sigmoid activation
- binary cross-entropy loss
- gradient computation
- gradient descent optimization
- probability prediction
- binary classification prediction
- custom evaluation metrics
- confusion matrix implementation

The model was implemented using an object-oriented design similar to the scikit-learn estimator API.

### Evaluation Results
- Accuracy: 0.791
- Precision: 0.738
- Recall: 0.706
- F1 Score: 0.722

The model achieved solid baseline performance using a relatively small feature set and establishes a reference point for the later NumPy-based implementation.

## Testing

Unit tests were added for:
- preprocessing utilities
- logistic regression model behavior
- gradient computation
- prediction logic
- custom evaluation metrics

The tests validate both numerical correctness and expected model behavior on small synthetic datasets.

## Conclusion

Week 1 established the foundations of the machine learning workflow through exploratory data analysis, preprocessing, model training, evaluation, and testing.

In addition to building a logistic regression baseline with scikit-learn, the model was also implemented completely from scratch using NumPy to better understand the mathematical foundations behind binary classification and gradient descent optimization.

The project now includes:
- reproducible preprocessing utilities
- a custom logistic regression implementation
- custom evaluation metrics
- automated unit tests
- CI integration
- structured project organization