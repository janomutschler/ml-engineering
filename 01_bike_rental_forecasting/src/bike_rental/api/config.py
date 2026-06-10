"""Runtime configuration for the prediction API, read from the environment."""

import os
from dataclasses import dataclass

from bike_rental.defs.resources.lakefs import LakeFSResource


@dataclass(frozen=True)
class ApiSettings:
    """Connection settings for the prediction API.

    Values are read from the same environment variables the Dagster pipeline
    uses, so the API and pipeline share one source of truth for where MLflow
    and LakeFS live.
    """

    mlflow_tracking_uri: str
    registered_model_name: str
    lakefs_host: str
    lakefs_access_key: str
    lakefs_secret_key: str
    lakefs_repository: str

    @classmethod
    def from_env(cls) -> "ApiSettings":
        """Build settings from environment variables, failing loudly if unset."""

        def required(key: str) -> str:
            value = os.environ.get(key)
            if not value:
                raise RuntimeError(f"Required environment variable {key} is not set.")
            return value

        return cls(
            mlflow_tracking_uri=required("MLFLOW_TRACKING_URI"),
            registered_model_name=os.environ.get("REGISTERED_MODEL_NAME", "bike_rental_forecaster"),
            lakefs_host=required("LAKEFS_HOST"),
            lakefs_access_key=required("LAKEFS_ACCESS_KEY"),
            lakefs_secret_key=required("LAKEFS_SECRET_KEY"),
            lakefs_repository=os.environ.get("LAKEFS_REPOSITORY", "bike-rental"),
        )

    def lakefs(self) -> LakeFSResource:
        """Build a LakeFSResource for reading published data from ``main``."""
        return LakeFSResource(
            host=self.lakefs_host,
            access_key=self.lakefs_access_key,
            secret_key=self.lakefs_secret_key,
            repository=self.lakefs_repository,
        )
