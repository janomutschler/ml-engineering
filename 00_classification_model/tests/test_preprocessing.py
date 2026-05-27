import numpy as np
import pandas as pd

from classification_model.data.preprocessing import (
    standardize_data,
    train_test_split_numpy,
)


def test_train_test_split_numpy_returns_expected_sizes():
    X = pd.DataFrame(
        {
            "feature_1": [1, 2, 3, 4, 5],
            "feature_2": [10, 20, 30, 40, 50],
        }
    )
    y = np.array([0, 1, 0, 1, 0])

    X_train, X_test, y_train, y_test = train_test_split_numpy(
        X,
        y,
        test_size=0.4,
        random_state=42,
    )

    assert len(X_train) == 3
    assert len(X_test) == 2
    assert len(y_train) == 3
    assert len(y_test) == 2


def test_train_test_split_numpy_is_reproducible():
    X = pd.DataFrame({"feature": [1, 2, 3, 4, 5]})
    y = np.array([0, 1, 0, 1, 0])

    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split_numpy(
        X,
        y,
        test_size=0.4,
        random_state=42,
    )

    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split_numpy(
        X,
        y,
        test_size=0.4,
        random_state=42,
    )

    assert X_train_1.equals(X_train_2)
    assert X_test_1.equals(X_test_2)

    np.testing.assert_array_equal(y_train_1, y_train_2)
    np.testing.assert_array_equal(y_test_1, y_test_2)


def test_standardize_data_uses_train_statistics():
    X_train = pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0],
            "fare": [10.0, 20.0, 30.0],
        }
    )
    X_test = pd.DataFrame(
        {
            "age": [50.0],
            "fare": [40.0],
        }
    )

    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)

    np.testing.assert_allclose(X_train_scaled.mean(axis=0), [0.0, 0.0])
    np.testing.assert_allclose(X_train_scaled.std(axis=0), [1.0, 1.0])

    assert X_test_scaled.shape == X_test.shape
