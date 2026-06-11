"""Dagster asset for promoting models to champion via MLflow aliases.

The promotion *policy* (``evaluate_promotion``) is a pure function of two
metric values, and the registry *orchestration* (``run_promotion``) takes the
MLflow client as a parameter. Both are unit-testable without Dagster: the
policy needs no mocks, and the orchestration needs only a small fake client.
The asset is a thin wrapper that builds the real client and logs the result.
"""

from dataclasses import dataclass

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


@dataclass(frozen=True)
class PromotionDecision:
    """Outcome of the promotion policy for one candidate version.

    Attributes
    ----------
    promote : bool
        Whether the candidate becomes the new champion.
    alias : str
        The alias to assign the candidate (``champion`` or ``challenger``).
    improvement : float | None
        Signed improvement over the champion on the configured metric, oriented
        so positive always means "better". ``None`` when there is no incumbent
        champion (bootstrap).

    """

    promote: bool
    alias: str
    improvement: float | None


def evaluate_promotion(
    new_metric: float,
    champion_metric: float | None,
    higher_is_better: bool,
    min_improvement: float,
) -> PromotionDecision:
    """Decide whether a candidate should become champion. Pure function.

    Parameters
    ----------
    new_metric : float
        The candidate version's metric value.
    champion_metric : float | None
        The current champion's metric value, or ``None`` if there is none.
    higher_is_better : bool
        Whether a larger metric value is better (e.g. R²). When ``False`` the
        improvement is computed in the opposite direction (e.g. RMSE).
    min_improvement : float
        The candidate is promoted only if it beats the champion by strictly
        more than this margin.

    Returns
    -------
    PromotionDecision
        The promotion outcome (promote flag, alias, signed improvement).

    """
    if champion_metric is None:
        return PromotionDecision(promote=True, alias=CHAMPION_ALIAS, improvement=None)

    improvement = new_metric - champion_metric if higher_is_better else champion_metric - new_metric

    if improvement > min_improvement:
        return PromotionDecision(promote=True, alias=CHAMPION_ALIAS, improvement=improvement)

    return PromotionDecision(promote=False, alias=CHALLENGER_ALIAS, improvement=improvement)


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


def run_promotion(
    client: MlflowClient,
    name: str,
    new_version: str,
    config: PromotionConfig,
) -> tuple[bool, str, dict]:
    """Read metrics, apply the promotion policy, and move the registry alias.

    Compares ``new_version`` against the current ``@champion`` on the configured
    metric (read from each version's source run) and assigns ``@champion`` or
    ``@challenger`` accordingly. The client is injected so this is testable with
    a fake registry.

    Parameters
    ----------
    client : MlflowClient
        MLflow registry client.
    name : str
        Registered model name.
    new_version : str
        The newly registered candidate version.
    config : PromotionConfig
        Promotion gate policy (metric, direction, minimum improvement).

    Returns
    -------
    tuple[bool, str, dict]
        The promote flag, a human-readable decision summary, and the output
        metadata dict to attach to the asset.

    """
    new_metric = _metric_for_version(client, name, new_version, config.metric)

    try:
        champion = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
    except MlflowException:
        champion = None

    champion_version = champion.version if champion is not None else None
    champion_metric = (
        _metric_for_version(client, name, champion_version, config.metric)
        if champion is not None
        else None
    )

    decision = evaluate_promotion(
        new_metric=new_metric,
        champion_metric=champion_metric,
        higher_is_better=config.higher_is_better,
        min_improvement=config.min_improvement,
    )

    client.set_registered_model_alias(name, decision.alias, new_version)

    if champion is None:
        summary = "promoted (bootstrap: no existing champion)"
    elif decision.promote:
        summary = (
            f"promoted ({config.metric}={decision.improvement:+.4f} > {config.min_improvement})"
        )
    else:
        summary = (
            "not promoted "
            f"(Δ{config.metric}={decision.improvement:+.4f} ≤ {config.min_improvement}); "
            f"champion remains v{champion_version}"
        )

    metadata = {
        "promoted": decision.promote,
        "decision": summary,
        "metric": config.metric,
        "new_version": new_version,
        "new_metric": new_metric,
        "champion_version": str(champion_version) if champion_version else "none",
        "champion_metric": champion_metric if champion_metric is not None else "none",
        "improvement": decision.improvement
        if decision.improvement is not None
        else "n/a (bootstrap)",
    }

    return decision.promote, summary, metadata


@asset
def model_promotion(
    context: AssetExecutionContext,
    trained_forecasting_model: str,
    mlflow_tracking: MlflowResource,
    config: PromotionConfig,
) -> None:
    """Promote the newly registered model version to champion if it wins.

    Compares the freshly registered version against the current ``@champion``
    on the configured metric. The ``@champion`` alias is moved to the new
    version only if it improves on the champion by more than ``min_improvement``
    (or if there is no champion yet); otherwise the new version is recorded as
    ``@challenger``. Production and inference always load
    ``models:/<name>@champion``.

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

    _, summary, metadata = run_promotion(
        client,
        mlflow_tracking.registered_model_name,
        trained_forecasting_model,
        config,
    )

    context.log.info("Version %s %s.", trained_forecasting_model, summary)
    context.add_output_metadata(metadata)
