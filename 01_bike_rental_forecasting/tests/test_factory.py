"""Tests for the model factory."""

import numpy as np
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from bike_rental.defs.training.factory import build_model


def test_build_model_returns_expected_estimators():
    """It maps each supported model type to the right estimator."""
    assert isinstance(build_model("xgboost"), XGBRegressor)
    assert isinstance(build_model("random_forest"), RandomForestRegressor)
    assert isinstance(build_model("dummy_mean"), DummyRegressor)


def test_linear_regression_is_wrapped_in_a_scaling_pipeline():
    """It scales features for linear regression to avoid scale sensitivity."""
    model = build_model("linear_regression")

    assert isinstance(model, Pipeline)
    assert "scaler" in model.named_steps


def test_log_target_wraps_the_estimator():
    """It wraps the estimator so training uses log1p and predictions use expm1."""
    model = build_model("random_forest", log_target=True)

    assert isinstance(model, TransformedTargetRegressor)
    assert model.func is np.log1p
    assert model.inverse_func is np.expm1


def test_unknown_model_type_raises():
    """It fails loudly on an unsupported model type."""
    with pytest.raises(ValueError, match="Unknown model_type"):
        build_model("nonexistent")
