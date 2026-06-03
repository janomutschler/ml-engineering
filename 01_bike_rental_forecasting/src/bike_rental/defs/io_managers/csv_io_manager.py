"""Local CSV IO manager for persisting pandas DataFrame assets."""

from pathlib import Path

import pandas as pd
from dagster import ConfigurableIOManager, InputContext, OutputContext


class LocalCsvIOManager(ConfigurableIOManager):
    """Persist pandas DataFrame assets as local CSV files."""

    base_path: str = "data/processed"
    project_root: str = "."

    @property
    def resolved_base_path(self) -> Path:
        """Return the absolute path to the local asset storage directory."""
        base_path = Path(self.base_path)

        if base_path.is_absolute():
            return base_path

        return Path(self.project_root).resolve() / base_path

    def _get_path(
        self,
        context: OutputContext | InputContext,
    ) -> Path:
        """Build the local CSV path for an asset."""
        asset_name = context.asset_key.path[-1]
        return self.resolved_base_path / f"{asset_name}.csv"

    def handle_output(
        self,
        context: OutputContext,
        obj: pd.DataFrame,
    ) -> None:
        """Persist a DataFrame asset to a local CSV file."""
        path = self._get_path(context)
        path.parent.mkdir(parents=True, exist_ok=True)

        obj.to_csv(path, index=False)

        context.log.info(
            "Wrote asset '%s' with %s rows and %s columns to %s",
            context.asset_key.to_user_string(),
            len(obj),
            len(obj.columns),
            path,
        )

        context.add_output_metadata(
            {
                "output_path": str(path),
                "rows": len(obj),
                "columns": len(obj.columns),
            }
        )

    def load_input(
        self,
        context: InputContext,
    ) -> pd.DataFrame:
        """Load a persisted local CSV asset as a DataFrame."""
        path = self._get_path(context)

        context.log.info(
            "Loading upstream asset '%s' from %s",
            context.asset_key.to_user_string(),
            path,
        )

        df = pd.read_csv(path)

        metadata = context.upstream_output.definition_metadata
        datetime_columns = metadata.get("datetime_columns", [])

        for column in datetime_columns:
            df[column] = pd.to_datetime(df[column], errors="raise")

        context.log.info(
            "Loaded upstream asset '%s' with %s rows and %s columns",
            context.asset_key.to_user_string(),
            len(df),
            len(df.columns),
        )

        return df
