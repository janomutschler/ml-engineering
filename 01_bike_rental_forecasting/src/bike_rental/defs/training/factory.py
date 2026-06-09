"""Model factory for the bike rental forecasting pipeline."""

from typing import Any

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

SUPPORTED_MODELS = ("xgboost", "random_forest", "linear_regression", "dummy_mean")


def _base_estimator(
    model_type: str,
    params: dict[str, Any],
    random_state: int,
) -> BaseEstimator:
    """Build the bare estimator for a model type.

    Linear regression is wrapped in a scaling pipeline so that the scaler is
    fit on the training fold only, which prevents leakage from the test fold.
    Tree-based models do not require scaling.
    """
    if model_type == "xgboost":
        return XGBRegressor(random_state=random_state, **params)
    if model_type == "lightgbm":
        return LGBMRegressor(random_state=random_state, **params)
    if model_type == "random_forest":
        return RandomForestRegressor(random_state=random_state, **params)
    if model_type == "linear_regression":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LinearRegression(**params)),
            ]
        )
    if model_type == "dummy_mean":
        return DummyRegressor(strategy="mean")

    raise ValueError(
        f"Unknown model_type {model_type!r}. Supported models: {', '.join(SUPPORTED_MODELS)}."
    )


def build_model(
    model_type: str,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
    log_target: bool = False,
) -> BaseEstimator:
    """Build a forecasting estimator from configuration.

    Hyperparameter defaults are owned by the resource's per-model ``Config``
    blocks, not this factory; ``params`` is expected to already contain the
    fully-resolved set for the chosen ``model_type``.

    Parameters
    ----------
    model_type : str
        One of ``SUPPORTED_MODELS``.
    params : dict[str, Any] | None, default=None
        Fully-resolved hyperparameters passed to the estimator constructor.
    random_state : int, default=42
        Seed applied to estimators that accept one.
    log_target : bool, default=False
        If True, wrap the estimator in a ``TransformedTargetRegressor`` that
        trains on ``log1p(target)`` and inverts predictions with ``expm1``.
        Safe for tree models; pathological for unregularized linear models.

    Returns
    -------
    BaseEstimator
        A scikit-learn compatible estimator ready to be fit.

    """
    estimator = _base_estimator(model_type, params or {}, random_state)

    if log_target:
        return TransformedTargetRegressor(
            regressor=estimator,
            func=np.log1p,
            inverse_func=np.expm1,
        )

    return estimator
