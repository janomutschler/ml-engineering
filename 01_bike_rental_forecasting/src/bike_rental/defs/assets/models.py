"""Dagster asset for training, evaluating, and registering the forecasting model."""

import mlflow
import mlflow.sklearn
import pandas as pd
from dagster import AssetExecutionContext, MetadataValue, asset
from mlflow.models import infer_signature

from bike_rental.defs.resources.mlflow import MlflowResource
from bike_rental.defs.resources.training_config import TrainingConfigResource
from bike_rental.defs.training.metrics import regression_metrics
from bike_rental.defs.utils.git import get_git_commit


@asset
def trained_forecasting_model(
    context: AssetExecutionContext,
    modeling_feature_set: pd.DataFrame,
    training_config: TrainingConfigResource,
    mlflow_tracking: MlflowResource,
) -> str:
    """Train, evaluate, and register the forecasting model, tracked in MLflow.

    The chronological split is derived in-process from the shared training
    configuration. The model is fit on the training period, scored on the
    hold-out test period, and logged to MLflow with its parameters, hold-out
    metrics, signature, input example, and run provenance (Dagster run id and
    git commit). The model is registered as a new version of the configured
    registered model.

    Parameters
    ----------
    context : AssetExecutionContext
        Dagster execution context.
    modeling_feature_set : pd.DataFrame
        Modeling-ready dataset containing engineered forecasting features.
    training_config : TrainingConfigResource
        Shared training configuration (split definition and model selection).
    mlflow_tracking : MlflowResource
        MLflow connection and run lifecycle.

    Returns
    -------
    str
        The MLflow model URI of the logged model.

    """
    split = training_config.split(modeling_feature_set)
    model = training_config.build_model()

    run_tags = {
        "dagster_run_id": context.run_id,
        "dagster_asset": context.asset_key.to_user_string(),
    }
    git_commit = get_git_commit()
    if git_commit:
        run_tags["git_commit"] = git_commit

    with mlflow_tracking.run(run_name=training_config.model_type, tags=run_tags) as active_run:
        mlflow.log_params(training_config.mlflow_params())

        model.fit(split.X_train, split.y_train)
        predictions = model.predict(split.X_test)
        metrics = regression_metrics(split.y_test, predictions)

        mlflow.log_metrics(
            {
                "holdout_mae": metrics["mae"],
                "holdout_rmse": metrics["rmse"],
                "holdout_rmsle": metrics["rmsle"],
                "holdout_r2": metrics["r2"],
                "training_rows": len(split.X_train),
                "test_rows": len(split.X_test),
            }
        )

        signature = infer_signature(split.X_train, model.predict(split.X_train))
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name=training_config.model_type,
            signature=signature,
            input_example=split.X_train.head(),
            registered_model_name=mlflow_tracking.registered_model_name,
        )

        run_id = active_run.info.run_id

    registered_version = getattr(model_info, "registered_model_version", None)

    context.log.info(
        "Logged %s to MLflow run %s (registered version %s); hold-out R² %.3f.",
        training_config.model_type,
        run_id,
        registered_version,
        metrics["r2"],
    )

    context.add_output_metadata(
        {
            "mlflow_run_id": run_id,
            "model_uri": model_info.model_uri,
            "registered_model": mlflow_tracking.registered_model_name,
            "registered_version": MetadataValue.text(str(registered_version)),
            "model_type": training_config.model_type,
            "log_target": training_config.log_target,
            "holdout_mae": metrics["mae"],
            "holdout_rmse": metrics["rmse"],
            "holdout_rmsle": metrics["rmsle"],
            "holdout_r2": metrics["r2"],
        }
    )

    return model_info.model_uri
