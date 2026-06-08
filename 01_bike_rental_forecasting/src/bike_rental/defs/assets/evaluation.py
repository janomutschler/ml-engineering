"""Dagster asset for evaluating forecasting model performance."""

import pandas as pd
from dagster import AssetExecutionContext, asset
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from xgboost import XGBRegressor

from bike_rental.defs.resources.training_config import TrainingConfigResource
from bike_rental.defs.utils.metadata import build_dataframe_metadata


@asset(io_manager_key="csv_io_manager")
def model_evaluation_metrics(
    context: AssetExecutionContext,
    modeling_feature_set: pd.DataFrame,
    trained_forecasting_model: XGBRegressor,
    training_config: TrainingConfigResource,
) -> pd.DataFrame:
    """Evaluate the trained model on the chronological hold-out test split.

    The test split is re-derived deterministically from ``modeling_feature_set``
    using the same shared ``training_config`` resource as the training asset.
    This guarantees the model is scored on the exact rows it did not see during
    training, without persisting an intermediate test dataset.

    Parameters
    ----------
    context : AssetExecutionContext
        Dagster execution context.
    modeling_feature_set : pd.DataFrame
        Modeling-ready dataset containing engineered forecasting features.
    trained_forecasting_model : XGBRegressor
        The fitted forecasting model.
    training_config : TrainingConfigResource
        Shared training configuration (split definition and hyperparameters).

    Returns
    -------
    pd.DataFrame
        Single-row evaluation metrics for the trained model.

    """
    split = training_config.split(modeling_feature_set)

    predictions = trained_forecasting_model.predict(split.X_test)

    metrics = {
        "model": type(trained_forecasting_model).__name__,
        "mae": mean_absolute_error(split.y_test, predictions),
        "rmse": root_mean_squared_error(split.y_test, predictions),
        "r2": r2_score(split.y_test, predictions),
        "test_rows": len(split.X_test),
    }

    metrics_df = pd.DataFrame([metrics])

    context.log.info(
        "Evaluated model with MAE %.2f, RMSE %.2f, and R² %.3f.",
        metrics["mae"],
        metrics["rmse"],
        metrics["r2"],
    )

    context.add_output_metadata(
        build_dataframe_metadata(
            metrics_df,
            extra_metadata=metrics,
        )
    )

    return metrics_df
