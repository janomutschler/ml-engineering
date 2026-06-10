"""Dagster asset for promoting models to champion via MLflow aliases."""

from dagster import AssetExecutionContext, Config, asset
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from bike_rental.defs.resources.mlflow import MlflowResource

CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"


class PromotionConfig(Config):
    """Policy for the champion/challenger promotion gate."""

    metric: str = "mean_r2"
    higher_is_better: bool = True
    min_improvement: float = 0.0


def _metric_for_version(
    client: MlflowClient,
    name: str,
    version: str,
    metric: str,
) -> float:
    """Read a metric value from the source run of a registered model version."""
    model_version = client.get_model_version(name, version)
    run = client.get_run(model_version.run_id)
    return run.data.metrics[metric]


@asset
def model_promotion(
    context: AssetExecutionContext,
    trained_forecasting_model: str,
    mlflow_tracking: MlflowResource,
    config: PromotionConfig,
) -> None:
    """Promote the newly registered model version to champion if it wins.

    Compares the freshly registered version against the current ``@champion``
    on the configured metric (read from each version's source run). The
    ``@champion`` alias is moved to the new version only if it improves on the
    champion by more than ``min_improvement`` (or if there is no champion yet).
    Otherwise the new version is recorded as ``@challenger``. Production and
    inference code always load ``models:/<name>@champion``.

    Parameters
    ----------
    context : AssetExecutionContext
        Dagster execution context.
    trained_forecasting_model : str
        Registered version number of the newly trained model (upstream asset).
    mlflow_tracking : MlflowResource
        MLflow connection and registry configuration.
    config : PromotionConfig
        Promotion gate policy (metric, direction, minimum improvement).

    """
    client = MlflowClient(tracking_uri=mlflow_tracking.tracking_uri)
    name = mlflow_tracking.registered_model_name
    new_version = trained_forecasting_model

    new_metric = _metric_for_version(client, name, new_version, config.metric)

    try:
        champion = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
    except MlflowException:
        champion = None

    if champion is None:
        client.set_registered_model_alias(name, CHAMPION_ALIAS, new_version)
        promoted, champion_version, champion_metric, improvement = True, None, None, None
        decision = "promoted (bootstrap: no existing champion)"
    else:
        champion_version = champion.version
        champion_metric = _metric_for_version(client, name, champion_version, config.metric)
        improvement = (
            new_metric - champion_metric
            if config.higher_is_better
            else champion_metric - new_metric
        )
        if improvement > config.min_improvement:
            client.set_registered_model_alias(name, CHAMPION_ALIAS, new_version)
            promoted = True
            decision = f"promoted ({config.metric}={improvement:+.4f} > {config.min_improvement})"
        else:
            client.set_registered_model_alias(name, CHALLENGER_ALIAS, new_version)
            promoted = False
            decision = (
                f"not promoted (Δ{config.metric}={improvement:+.4f} ≤ {config.min_improvement}); "
                f"champion remains v{champion_version}"
            )

    context.log.info("Version %s %s.", new_version, decision)

    context.add_output_metadata(
        {
            "promoted": promoted,
            "decision": decision,
            "metric": config.metric,
            "new_version": new_version,
            "new_metric": new_metric,
            "champion_version": str(champion_version) if champion_version else "none",
            "champion_metric": champion_metric if champion_metric is not None else "none",
            "improvement": improvement if improvement is not None else "n/a (bootstrap)",
        }
    )
