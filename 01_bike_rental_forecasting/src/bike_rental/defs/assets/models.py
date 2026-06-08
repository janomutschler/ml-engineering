"""Dagster asset for training and evaluating the bike rental forecasting model."""

import pandas as pd
from dagster import AssetExecutionContext, asset
from sklearn.base import BaseEstimator

from bike_rental.defs.resources.training_config import TrainingConfigResource
from bike_rental.defs.training.metrics import regression_metrics


@asset
def trained_forecasting_model(
    context: AssetExecutionContext,
    modeling_feature_set: pd.DataFrame,
    training_config: TrainingConfigResource,
) -> BaseEstimator:
    """Train and evaluate the forecasting model on a chronological split.

    The train/test split is derived in-process from ``modeling_feature_set``
    using the shared ``training_config`` resource, so no intermediate split
    datasets are persisted. The model is fit on the training period, scored on
    the hold-out test period, and the hold-out metrics are attached as output
    metadata. The fitted model is returned as the asset value and persisted by
    the default IO manager.

    Parameters
    ----------
    context : AssetExecutionContext
        Dagster execution context.
    modeling_feature_set : pd.DataFrame
        Modeling-ready dataset containing engineered forecasting features.
    training_config : TrainingConfigResource
        Shared training configuration (split definition and model selection).

    Returns
    -------
    BaseEstimator
        The fitted forecasting model.

    """
    split = training_config.split(modeling_feature_set)

    model = training_config.build_model()
    model.fit(split.X_train, split.y_train)

    predictions = model.predict(split.X_test)
    metrics = regression_metrics(split.y_test, predictions)

    context.log.info(
        "Trained %s (log_target=%s) on %s rows; hold-out MAE %.2f, RMSE %.2f, RMSLE %.3f, R² %.3f.",
        training_config.model_type,
        training_config.log_target,
        len(split.X_train),
        metrics["mae"],
        metrics["rmse"],
        metrics["rmsle"],
        metrics["r2"],
    )

    context.add_output_metadata(
        {
            "model_type": training_config.model_type,
            "log_target": training_config.log_target,
            "training_rows": len(split.X_train),
            "test_rows": len(split.X_test),
            "feature_count": len(training_config.feature_columns),
            "train_ratio": training_config.train_ratio,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "rmsle": metrics["rmsle"],
            "r2": metrics["r2"],
        }
    )

    return model
