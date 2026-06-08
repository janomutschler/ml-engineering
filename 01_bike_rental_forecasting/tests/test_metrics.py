"""Tests for regression metrics."""

from bike_rental.defs.training.metrics import regression_metrics


def test_perfect_predictions_yield_zero_error():
    """It reports zero error and perfect R² for exact predictions."""
    y_true = [0, 10, 20, 30]

    metrics = regression_metrics(y_true, y_true)

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["rmsle"] == 0.0
    assert metrics["r2"] == 1.0


def test_rmsle_clips_negative_predictions():
    """It clips negative predictions so RMSLE stays defined."""
    y_true = [1, 2, 3]
    y_pred = [-5.0, 2.0, 3.0]

    metrics = regression_metrics(y_true, y_pred)

    assert metrics["rmsle"] >= 0.0
