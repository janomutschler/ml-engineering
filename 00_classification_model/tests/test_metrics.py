import numpy as np
from evaluation.metrics import (
    accuracy,
    confusion_matrix,
    f1_score,
    precision,
    recall,
)


def test_accuracy_returns_expected_value():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])

    assert accuracy(y_true, y_pred) == 0.75


def test_precision_returns_expected_value():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])

    assert precision(y_true, y_pred) == 1.0


def test_recall_returns_expected_value():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])

    assert recall(y_true, y_pred) == 0.5


def test_f1_score_returns_expected_value():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])

    assert f1_score(y_true, y_pred) == 2 / 3


def test_confusion_matrix_returns_expected_layout():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1])

    result = confusion_matrix(y_true, y_pred)

    expected = np.array(
        [
            [1, 1],
            [1, 1],
        ]
    )

    np.testing.assert_array_equal(result, expected)


def test_precision_returns_zero_when_no_predicted_positives():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 0])

    assert precision(y_true, y_pred) == 0.0


def test_recall_returns_zero_when_no_actual_positives():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0])

    assert recall(y_true, y_pred) == 0.0


def test_f1_score_returns_zero_when_precision_and_recall_are_zero():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0])

    assert f1_score(y_true, y_pred) == 0.0
