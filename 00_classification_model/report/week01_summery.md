# Week 1 - Classification Summary

## Objective
Explore the Titanic dataset and build baseline binary classification models to predict passenger survival using logistic regression.

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
- AUC: 0.852

The model achieved strong baseline performance while maintaining balanced classification behavior across the evaluation metrics. The ROC AUC score further indicates that the model separates survival and non-survival cases effectively.

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
- ROC curve and AUC evaluation

The model was implemented using an object-oriented design inspired by the scikit-learn estimator API.

### Evaluation Results
- Accuracy: 0.836
- Precision: 0.820
- Recall: 0.735
- F1 Score: 0.775
- AUC: 0.870

## Model Comparison

The custom NumPy implementation was compared against the scikit-learn implementation using the same train-test split and feature scaling pipeline. (in the Baseline model the sklearn train-test split was used thats why the sklearn model archives different metrics here then in Logistic Regression Baseline) 

| Metric | NumPy | scikit-learn |
|---|---|---|
| Accuracy | 0.836 | 0.831 |
| Precision | 0.820 | 0.797 |
| Recall | 0.735 | 0.750 |
| F1 Score | 0.775 | 0.773 |
| AUC | 0.870 | 0.871 |

The NumPy implementation achieved very similar overall performance to the scikit-learn model, including nearly identical ROC AUC scores. The comparison between the learned weights and bias values also showed that both implementations converged toward very similar model parameters.

The NumPy implementation achieved slightly higher accuracy, precision, and F1 score, while the scikit-learn model achieved slightly higher recall and AUC. Overall, the results demonstrate that the custom implementation approximates the behavior of the optimized scikit-learn implementation closely while still exposing the internal learning mechanics of logistic regression.

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

Compared to the scikit-learn baseline, the NumPy implementation achieved very similar overall behavior and evaluation performance, including nearly identical ROC AUC scores. The comparison of the learned weights and bias values also showed that both implementations converged toward very similar model parameters.

At the same time, implementing logistic regression from scratch exposed the internal mechanics of the algorithm, including probability prediction, sigmoid activation, loss computation, gradient calculation, and parameter updates through gradient descent. This helped build a deeper understanding of how machine learning models learn from data beyond using high-level library abstractions.

The project now includes:
- reproducible preprocessing utilities
- a custom logistic regression implementation
- custom evaluation metrics
- ROC curve and AUC evaluation
- automated unit tests
- CI integration
- structured project organization
- comparison between library-based and custom model implementations