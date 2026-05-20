import numpy as np


def accuracy(y_true, y_pred):
    """
    Compute the share of correctly predicted labels.
    """
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred):
    """
    Compute precision: true positives divided by all predicted positives.
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives = np.sum(y_pred == 1)

    if predicted_positives == 0:
        return 0.0

    return true_positives / predicted_positives


def recall(y_true, y_pred):
    """
    Compute recall: true positives divided by all actual positives.
    """
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1)

    if actual_positives == 0:
        return 0.0

    return true_positives / actual_positives


def f1_score(y_true, y_pred):
    """
    Compute F1 score as the harmonic mean of precision and recall.
    """
    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)

    if precision_value + recall_value == 0:
        return 0.0

    return 2 * (precision_value * recall_value) / (precision_value + recall_value)


def confusion_matrix(y_true, y_pred):
    """
    Compute a binary confusion matrix.

    Returns:
        [[true_negatives, false_positives],
         [false_negatives, true_positives]]
    """
    true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    true_positives = np.sum((y_true == 1) & (y_pred == 1))

    return np.array(
        [
            [true_negatives, false_positives],
            [false_negatives, true_positives],
        ]
    )
