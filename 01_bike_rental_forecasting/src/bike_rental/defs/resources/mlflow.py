"""MLflow tracking resource for the bike rental forecasting pipeline."""

from collections.abc import Iterator
from contextlib import contextmanager

import mlflow
from dagster import ConfigurableResource


class MlflowResource(ConfigurableResource):
    """Configure MLflow tracking and manage the run lifecycle.

    The resource owns the connection (tracking URI, experiment name, registry
    name); assets decide what to log within the yielded run. The tracking URI
    must point at a database-backed store (e.g. a server backed by SQLite or
    Postgres) for the model registry to work.

    """

    tracking_uri: str
    experiment_name: str = "bike_rental_forecasting"
    registered_model_name: str = "bike_rental_forecaster"

    @contextmanager
    def run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Iterator[mlflow.ActiveRun]:
        """Start an MLflow run against the configured experiment.

        The run is closed automatically on exit and marked FAILED if the body
        raises, so a failed training run is recorded as such in MLflow.

        Parameters
        ----------
        run_name : str | None, default=None
            Human-readable name for the run.
        tags : dict[str, str] | None, default=None
            Tags attached to the run (e.g. provenance metadata).

        Yields
        ------
        mlflow.ActiveRun
            The active MLflow run.

        """
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=run_name, tags=tags) as active_run:
            yield active_run
