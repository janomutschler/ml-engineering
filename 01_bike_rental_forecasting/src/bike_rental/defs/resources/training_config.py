"""Training configuration resource for the bike rental forecasting pipeline."""

from typing import Any

from dagster import Config, ConfigurableResource
from pydantic import Field
from sklearn.base import BaseEstimator

from bike_rental.defs.training.factory import build_model


class XGBoostParams(Config):
    """Tunable hyperparameters for the XGBoost forecasting model."""

    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    objective: str = "reg:squarederror"


class LightGBMParams(Config):
    """Tunable hyperparameters for the LightGBM forecasting model."""

    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = -1
    num_leaves: int = 31
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0


class RandomForestParams(Config):
    """Tunable hyperparameters for the random forest forecasting model."""

    n_estimators: int = 300
    min_samples_leaf: int = 2
    n_jobs: int = -1


class LinearRegressionParams(Config):
    """Tunable hyperparameters for the linear regression baseline."""

    fit_intercept: bool = True


class TrainingConfigResource(ConfigurableResource):
    """Single source of truth for the training and evaluation configuration.

    Per-model hyperparameters live in their own typed config blocks so each
    one renders as a discoverable field in the launchpad and a parameter from
    one model cannot leak into another. The active block is chosen by
    ``model_type`` at build time.

    """

    # Data definition (shared by backtest and final training).
    feature_columns: list[str]
    target_column: str
    time_column: str = "datetime_hour"

    # Evaluation.
    n_splits: int = 5

    # Model selection.
    model_type: str = "xgboost"
    xgboost: XGBoostParams = Field(default_factory=XGBoostParams)
    lightgbm: LightGBMParams = Field(default_factory=LightGBMParams)
    random_forest: RandomForestParams = Field(default_factory=RandomForestParams)
    linear_regression: LinearRegressionParams = Field(default_factory=LinearRegressionParams)

    random_state: int = 42
    log_target: bool = False

    def active_params(self) -> dict[str, Any]:
        """Return the hyperparameter block matching ``model_type``."""
        blocks = {
            "xgboost": self.xgboost,
            "lightgbm": self.lightgbm,
            "random_forest": self.random_forest,
            "linear_regression": self.linear_regression,
            "dummy_mean": None,
        }
        if self.model_type not in blocks:
            raise ValueError(f"Unknown model_type: {self.model_type!r}.")

        block = blocks[self.model_type]
        return block.model_dump() if block is not None else {}

    def build_model(self) -> BaseEstimator:
        """Build the configured forecasting estimator."""
        return build_model(
            model_type=self.model_type,
            params=self.active_params(),
            random_state=self.random_state,
            log_target=self.log_target,
        )

    def mlflow_params(self) -> dict[str, Any]:
        """Flatten the configuration into a dict of MLflow params to log."""
        params: dict[str, Any] = {
            "model_type": self.model_type,
            "log_target": self.log_target,
            "n_splits": self.n_splits,
            "random_state": self.random_state,
            "feature_count": len(self.feature_columns),
            "target_column": self.target_column,
        }
        params.update(self.active_params())
        return params
