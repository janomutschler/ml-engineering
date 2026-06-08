"""Training configuration resource for the bike rental forecasting pipeline."""

import pandas as pd
from dagster import Config, ConfigurableResource
from pydantic import Field
from sklearn.base import BaseEstimator

from bike_rental.defs.training.factory import build_model
from bike_rental.defs.training.splitting import TrainTestSplit, chronological_split


class XGBoostParams(Config):
    """Tunable hyperparameters for the XGBoost forecasting model."""

    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    objective: str = "reg:squarederror"


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

    # Split definition (shared by training and evaluation).
    feature_columns: list[str]
    target_column: str
    time_column: str = "datetime_hour"
    train_ratio: float = 0.8

    # Model selection.
    model_type: str = "xgboost"
    xgboost: XGBoostParams = Field(default_factory=XGBoostParams)
    random_forest: RandomForestParams = Field(default_factory=RandomForestParams)
    linear_regression: LinearRegressionParams = Field(default_factory=LinearRegressionParams)

    random_state: int = 42
    log_target: bool = False

    def split(self, modeling_feature_set: pd.DataFrame) -> TrainTestSplit:
        """Derive the chronological train/test split for a modeling feature set."""
        return chronological_split(
            modeling_feature_set,
            feature_columns=self.feature_columns,
            target_column=self.target_column,
            time_column=self.time_column,
            train_ratio=self.train_ratio,
        )

    def _active_params(self) -> dict:
        """Select the hyperparameter block matching ``model_type``."""
        blocks = {
            "xgboost": self.xgboost,
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
            params=self._active_params(),
            random_state=self.random_state,
            log_target=self.log_target,
        )
