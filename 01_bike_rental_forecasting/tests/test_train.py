"""Tests for the extracted training core (backtest_and_fit) and metric aggregation."""

import numpy as np
import pandas as pd
import pytest

from bike_rental.defs.resources.training_config import TrainingConfigResource
from bike_rental.defs.training.metrics import aggregate_fold_metrics
from bike_rental.defs.training.train import backtest_and_fit

_METRICS = ("mae", "rmse", "rmsle", "r2")
_FEATURES = ["temperature_c", "hour_sin", "hour_cos", "lag_24h"]


def _modeling_frame(days: int = 30) -> pd.DataFrame:
    """Small modeling-ready frame with a daily demand shape and no NaNs."""
    n = days * 24
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    demand = np.clip(50 + 30 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 5, n), 0, None)
    return pd.DataFrame(
        {
            "datetime_hour": idx,
            "temperature_c": 15.0 + rng.normal(0, 3, n),
            "hour_sin": np.sin(2 * np.pi * idx.hour / 24),
            "hour_cos": np.cos(2 * np.pi * idx.hour / 24),
            "lag_24h": np.concatenate([np.zeros(24), demand[:-24]]),
            "total_rentals": demand,
        }
    )


def _config(**overrides) -> TrainingConfigResource:
    base = {
        "feature_columns": _FEATURES,
        "target_column": "total_rentals",
        "model_type": "linear_regression",
        "n_splits": 4,
    }
    base.update(overrides)
    return TrainingConfigResource(**base)


def test_backtest_and_fit_returns_one_metric_row_per_fold():
    """Stage 1 produces n_splits rows with every metric column populated."""
    result = backtest_and_fit(_modeling_frame(), _config(n_splits=4), _METRICS)

    assert len(result.fold_metrics) == 4
    for metric in _METRICS:
        assert metric in result.fold_metrics.columns
        assert result.fold_metrics[metric].notna().all()


def test_backtest_and_fit_aggregates_mean_and_std_per_metric():
    """Aggregates carry a mean and std entry for each metric."""
    result = backtest_and_fit(_modeling_frame(), _config(), _METRICS)

    for metric in _METRICS:
        assert f"mean_{metric}" in result.aggregates
        assert f"std_{metric}" in result.aggregates


def test_backtest_and_fit_returns_a_fitted_model_on_the_exact_feature_columns():
    """Stage 2 returns a fitted model and the float64 matrix it trained on."""
    result = backtest_and_fit(_modeling_frame(), _config(), _METRICS)

    assert list(result.X_all.columns) == _FEATURES
    assert all(dtype == np.float64 for dtype in result.X_all.dtypes)

    predictions = result.model.predict(result.X_all)
    assert len(predictions) == len(result.X_all)


def test_aggregate_fold_metrics_computes_mean_and_std():
    """The aggregator matches a hand-computed mean and std."""
    folds = pd.DataFrame({"r2": [0.8, 0.9], "mae": [10.0, 20.0]})

    aggregates = aggregate_fold_metrics(folds, ("r2", "mae"))

    assert aggregates["mean_r2"] == pytest.approx(0.85)
    assert aggregates["mean_mae"] == pytest.approx(15.0)
    # pandas std is sample (ddof=1) by default
    assert aggregates["std_mae"] == pytest.approx(folds["mae"].std())
