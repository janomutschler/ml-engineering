"""Dagster asset for backtesting, training, and registering the forecasting model."""

import mlflow
import mlflow.sklearn
import pandas as pd
from dagster import AssetExecutionContext, MetadataValue, asset
from mlflow.models import infer_signature

from bike_rental.defs.resources.mlflow import MlflowResource
from bike_rental.defs.resources.training_config import TrainingConfigResource
from bike_rental.defs.training.backtest import walk_forward_backtest
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

    The model build has two stages, recorded together so the registered model
    version links directly to the evidence that justifies it:

    1. Evaluate the recipe with a walk-forward backtest across ``n_splits``
       expanding folds. The per-fold and aggregate cross-validation metrics
       are the honest estimate of how this configuration generalizes
       over time.
    2. Produce the artifact by fitting the same configuration on all available
       data and registering it. The deployed model uses the full history; the
       backtest, not a held-out slice, is its performance estimate.

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
        The MLflow model URI of the registered model.

    """
    feature_columns = training_config.feature_columns
    target_column = training_config.target_column

    run_tags = {"dagster_run_id": context.run_id, "lakefs_commit": data_version}
    git_commit = get_git_commit()
    if git_commit:
        run_tags["git_commit"] = git_commit

    with mlflow_tracking.run(run_name=training_config.model_type, tags=run_tags) as active_run:
        mlflow.log_params(training_config.mlflow_params())

        # Stage 1: evaluate the recipe with a walk-forward backtest.
        fold_metrics = walk_forward_backtest(
            modeling_feature_set,
            make_model=training_config.build_model,
            feature_columns=feature_columns,
            target_column=target_column,
            time_column=training_config.time_column,
            n_splits=training_config.n_splits,
        )

        for _, fold in fold_metrics.iterrows():
            mlflow.log_metrics(
                {metric: float(fold[metric]) for metric in _METRICS},
                step=int(fold["fold"]),
            )

        aggregates = {f"mean_{metric}": float(fold_metrics[metric].mean()) for metric in _METRICS}
        aggregates.update(
            {f"std_{metric}": float(fold_metrics[metric].std()) for metric in _METRICS}
        )
        mlflow.log_metrics(aggregates)

        # Stage 2: fit the final model on all data and register it.
        ordered = modeling_feature_set.sort_values(training_config.time_column)
        X_all = ordered[feature_columns]
        y_all = ordered[target_column]

        final_model = training_config.build_model()
        final_model.fit(X_all, y_all)

        signature = infer_signature(X_all, final_model.predict(X_all))
        model_info = mlflow.sklearn.log_model(
            sk_model=final_model,
            name=training_config.model_type,
            signature=signature,
            input_example=X_all.head(),
            registered_model_name=mlflow_tracking.registered_model_name,
        )

        run_id = active_run.info.run_id

    registered_version = getattr(model_info, "registered_model_version", None)

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
            "backtest_folds": build_dataframe_metadata(fold_metrics)["preview"],
        }
    )

    return str(registered_version)
