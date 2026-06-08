"""Tests for chronological train/test splitting."""

import pandas as pd
import pytest

from bike_rental.defs.training.splitting import chronological_split


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime_hour": pd.to_datetime(
                [
                    "2024-01-01 03:00:00",
                    "2024-01-01 00:00:00",
                    "2024-01-01 02:00:00",
                    "2024-01-01 01:00:00",
                ]
            ),
            "f1": [30, 0, 20, 10],
            "total_rentals": [3, 0, 2, 1],
        }
    )


def test_chronological_split_orders_and_partitions_by_time():
    """It assigns the earliest rows to train and the latest to test."""
    split = chronological_split(
        _sample_frame(),
        feature_columns=["f1"],
        target_column="total_rentals",
        time_column="datetime_hour",
        train_ratio=0.5,
    )

    assert split.X_train["f1"].tolist() == [0, 10]
    assert split.X_test["f1"].tolist() == [20, 30]
    assert split.y_train.tolist() == [0, 1]
    assert split.y_test.tolist() == [2, 3]


def test_chronological_split_is_deterministic_regardless_of_input_order():
    """It produces the same split no matter how the input rows are ordered."""
    df = _sample_frame()

    ordered = chronological_split(
        df,
        feature_columns=["f1"],
        target_column="total_rentals",
        time_column="datetime_hour",
        train_ratio=0.5,
    )
    shuffled = chronological_split(
        df.sample(frac=1, random_state=1),
        feature_columns=["f1"],
        target_column="total_rentals",
        time_column="datetime_hour",
        train_ratio=0.5,
    )

    assert ordered.X_test["f1"].tolist() == shuffled.X_test["f1"].tolist()
    assert ordered.y_test.tolist() == shuffled.y_test.tolist()


def test_chronological_split_raises_on_missing_columns():
    """It fails loudly when a configured feature column is absent."""
    with pytest.raises(ValueError, match="missing required columns"):
        chronological_split(
            _sample_frame(),
            feature_columns=["f1", "does_not_exist"],
            target_column="total_rentals",
            time_column="datetime_hour",
            train_ratio=0.5,
        )
