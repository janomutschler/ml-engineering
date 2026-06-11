"""Dagster asset for backtesting, training, and registering the forecasting model."""

import mlflow
import mlflow.sklearn
import pandas as pd
from dagster import AssetExecutionContext, MetadataValue, asset
from mlflow.models import infer_signature

from bike_rental.defs.resources.mlflow import MlflowResource
from bike_rental.defs.resources.training_config import TrainingConfigResource
from bike_rental.defs.training.train import backtest_and_fit
from bike_rental.defs.utils.git import get_git_commit
from bike_rental.defs.utils.metadata import build_dataframe_metadata

_METRICS = ("mae", "rmse", "rmsle", "r2")


@asset
def trained_forecasting_model(
    context: AssetExecutionContext,
    modeling_feature_set: pd.DataFrame,
    data_version: str,
    training_config: TrainingConfigResource,
    mlflow_tracking: MlflowResource,
) -> str:
    """Backtest, train, and register the forecasting model in one MLflow run.

    The ML work — walk-forward backtest plus the final fit on all data — lives
    in :func:`backtest_and_fit`. This asset wraps that core in a single MLflow
    run so the registered model version links directly to the evidence that
    justifies it: the per-fold cross-validation metrics are the honest estimate
    of how the configuration generalizes, and the deployed artifact is fit on
    the full history.

    Parameters
    ----------
    context : AssetExecutionContext
        Dagster execution context.
    modeling_feature_set : pd.DataFrame
        Modeling-ready dataset containing engineered forecasting features.
    data_version : str
        LakeFS commit id of the data this run trains on (logged for lineage).
    training_config : TrainingConfigResource
        Shared training configuration (model selection and backtest settings).
    mlflow_tracking : MlflowResource
        MLflow connection and run lifecycle.

    Returns
    -------
    str
        The registered model version (as a string). The promotion gate consumes
        this to compare the new version against the current champion.

    """
    run_tags = {"dagster_run_id": context.run_id, "lakefs_commit": data_version}
    git_commit = get_git_commit()
    if git_commit:
        run_tags["git_commit"] = git_commit

    with mlflow_tracking.run(run_name=training_config.model_type, tags=run_tags) as active_run:
        mlflow.log_params(training_config.mlflow_params())
        mlflow.log_param("feature_columns", ",".join(training_config.feature_columns))

        result = backtest_and_fit(modeling_feature_set, training_config, _METRICS)

        for _, fold in result.fold_metrics.iterrows():
            mlflow.log_metrics(
                {metric: float(fold[metric]) for metric in _METRICS},
                step=int(fold["fold"]),
            )
        mlflow.log_metrics(result.aggregates)

        signature = infer_signature(result.X_all, result.model.predict(result.X_all))
        model_info = mlflow.sklearn.log_model(
            sk_model=result.model,
            name=training_config.model_type,
            signature=signature,
            input_example=result.X_all.head(),
            registered_model_name=mlflow_tracking.registered_model_name,
        )

        run_id = active_run.info.run_id

    registered_version = getattr(model_info, "registered_model_version", None)
    aggregates = result.aggregates

    context.log.info(
        "Backtested and registered %s (run %s, version %s): "
        "R² %.3f (±%.3f), MAE %.2f over %s folds.",
        training_config.model_type,
        run_id,
        registered_version,
        aggregates["mean_r2"],
        aggregates["std_r2"],
        aggregates["mean_mae"],
        training_config.n_splits,
    )

    context.add_output_metadata(
        {
            "mlflow_run_id": run_id,
            "lakefs_commit": data_version,
            "model_uri": model_info.model_uri,
            "registered_model": mlflow_tracking.registered_model_name,
            "registered_version": MetadataValue.text(str(registered_version)),
            "model_type": training_config.model_type,
            "n_splits": training_config.n_splits,
            "mean_r2": aggregates["mean_r2"],
            "std_r2": aggregates["std_r2"],
            "mean_mae": aggregates["mean_mae"],
            "mean_rmsle": aggregates["mean_rmsle"],
            "backtest_folds": build_dataframe_metadata(result.fold_metrics)["preview"],
        }
    )

    return str(registered_version)
