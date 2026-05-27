import numpy as np

from classification_model.evaluation.metrics import (
    accuracy,
    confusion_matrix,
    f1_score,
    precision,
    recall,
)
from classification_model.models.logistic_regression import LogisticRegressionScratch


def test_sigmoid_outputs_expected_values():
    model = LogisticRegressionScratch()

    result = model.sigmoid(np.array([0.0]))

    np.testing.assert_allclose(result, [0.5])


def test_predict_proba_returns_values_between_zero_and_one():
    model = LogisticRegressionScratch()
    model.w = np.array([1.0, -1.0])
    model.b = 0.0

    X = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    probabilities = model.predict_proba(X)

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)


def test_predict_returns_binary_labels():
    model = LogisticRegressionScratch(threshold=0.5)
    model.w = np.array([1.0])
    model.b = 0.0

    X = np.array(
        [
            [-10.0],
            [10.0],
        ]
    )

    predictions = model.predict(X)

    np.testing.assert_array_equal(predictions, [0, 1])


def test_compute_loss_is_small_for_good_predictions():
    model = LogisticRegressionScratch()

    y_true = np.array([1, 0])
    y_pred = np.array([0.99, 0.01])

    loss = model.compute_loss(y_true, y_pred)

    assert loss < 0.02


def test_compute_gradients_shapes_match_parameters():
    model = LogisticRegressionScratch()

    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    y = np.array([1, 0])
    y_pred = np.array([0.8, 0.2])

    dw, db = model.compute_gradients(X, y, y_pred)

    assert dw.shape == (2,)
    assert isinstance(db, np.float64) or isinstance(db, float)


def test_model_can_fit_simple_linearly_separable_data():
    X = np.array(
        [
            [-2.0],
            [-1.0],
            [1.0],
            [2.0],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)

    predictions = model.predict(X)

    np.testing.assert_array_equal(predictions, y)


def test_custom_metrics_return_expected_values():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])

    assert accuracy(y_true, y_pred) == 0.75
    assert precision(y_true, y_pred) == 1.0
    assert recall(y_true, y_pred) == 0.5
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
