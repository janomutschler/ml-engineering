"""Local Parquet IO manager for persisting pandas DataFrame assets."""

from pathlib import Path

import pandas as pd
from dagster import ConfigurableIOManager, InputContext, OutputContext


class LocalParquetIOManager(ConfigurableIOManager):
    """Persist pandas DataFrame assets as local Parquet files."""

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
        """Build the local Parquet path for an asset."""
        asset_name = context.asset_key.path[-1]
        return self.resolved_base_path / f"{asset_name}.parquet"

    def handle_output(
        self,
        context: OutputContext,
        obj: pd.DataFrame,
    ) -> None:
        """Persist a DataFrame asset to a local Parquet file."""
        path = self._get_path(context)
        path.parent.mkdir(parents=True, exist_ok=True)

        obj.to_parquet(path, index=False)

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
                "storage_format": "parquet",
            }
        )

    def load_input(
        self,
        context: InputContext,
    ) -> pd.DataFrame:
        """Load a persisted local Parquet asset as a DataFrame."""
        path = self._get_path(context)

        context.log.info(
            "Loading upstream asset '%s' from %s",
            context.asset_key.to_user_string(),
            path,
        )

        df = pd.read_parquet(path)

        if df.empty:
            raise ValueError(f"Loaded empty DataFrame from {path}")

        context.log.info(
            "Loaded upstream asset '%s' with %s rows and %s columns",
            context.asset_key.to_user_string(),
            len(df),
            len(df.columns),
        )

        return df
