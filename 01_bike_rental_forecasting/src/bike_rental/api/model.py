"""Load and cache the champion model from the MLflow registry."""

from dataclasses import dataclass

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.base import BaseEstimator

CHAMPION_ALIAS = "champion"


@dataclass
class ChampionModel:
    """The currently-served model and the lineage of where it came from.

    Attributes
    ----------
    model : BaseEstimator
        The fitted estimator loaded from the registry.
    feature_columns : list[str]
        Input column order taken from the logged model signature.
    version : str
        Registered model version behind the ``@champion`` alias.
    data_commit : str | None
        LakeFS commit the model was trained on (from the source run's tags).

    """

    model: BaseEstimator
    feature_columns: list[str]
    version: str
    data_commit: str | None


def load_champion(tracking_uri: str, model_name: str) -> ChampionModel:
    """Load the ``@champion`` model and its lineage from the registry.

    Parameters
    ----------
    tracking_uri : str
        MLflow tracking URI.
    model_name : str
        Registered model name.

    Returns
    -------
    ChampionModel
        The loaded model plus its version, feature columns, and data commit.

    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    model_uri = f"models:/{model_name}@{CHAMPION_ALIAS}"
    version = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    model = mlflow.sklearn.load_model(model_uri)

    run = client.get_run(version.run_id)
    data_commit = run.data.tags.get("lakefs_commit")

    model_info = mlflow.models.get_model_info(model_uri)
    feature_columns = model_info.signature.inputs.input_names()

    return ChampionModel(
        model=model,
        feature_columns=feature_columns,
        version=version.version,
        data_commit=data_commit,
    )
