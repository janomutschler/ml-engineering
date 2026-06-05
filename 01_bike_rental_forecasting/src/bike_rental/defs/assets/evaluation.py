"""Dagster assets for evaluating forecasting model performance."""

import joblib
import pandas as pd
from dagster import AssetExecutionContext, asset
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from bike_rental.defs.constants import TARGET_COLUMN
from bike_rental.defs.utils.metadata import build_dataframe_metadata


@asset(io_manager_key="csv_io_manager")
def model_evaluation_metrics(
    context: AssetExecutionContext,
    xgboost_forecasting_model: str,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate the trained forecasting model on the test dataset.

    Parameters
    ----------
    context : AssetExecutionContext
        Dagster execution context.
    xgboost_forecasting_model : str
        Path to the persisted trained forecasting model.
    X_test : pd.DataFrame
        Test feature dataset.
    y_test : pd.DataFrame
        Test target dataset.

    Returns
    -------
    pd.DataFrame
        Evaluation metrics for the trained forecasting model.

    """
    model = joblib.load(xgboost_forecasting_model)

    y_true = y_test[TARGET_COLUMN]
    predictions = model.predict(X_test)

    metrics = {
        "model": "XGBoost",
        "mae": mean_absolute_error(y_true, predictions),
        "rmse": root_mean_squared_error(y_true, predictions),
        "r2": r2_score(y_true, predictions),
        "test_rows": len(X_test),
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
            extra_metadata={
                "model": metrics["model"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "test_rows": metrics["test_rows"],
                "model_path": xgboost_forecasting_model,
            },
        )
    )

    return metrics_df
