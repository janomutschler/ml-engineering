"""CSV IO manager for persisting selected pandas DataFrame assets."""

from pathlib import Path

import pandas as pd
from dagster import ConfigurableIOManager, InputContext, OutputContext


class CsvIOManager(ConfigurableIOManager):
    """Persist selected DataFrame assets as CSV files."""

    base_path: str

    def _get_path(
        self,
        context: OutputContext | InputContext,
    ) -> Path:
        """Build the output CSV path for an asset."""
        asset_name = context.asset_key.path[-1]

        return Path(self.base_path) / f"{asset_name}.csv"

    def handle_output(
        self,
        context: OutputContext,
        obj: pd.DataFrame,
    ) -> None:
        """Persist a DataFrame asset to CSV."""
        path = self._get_path(context)

        path.parent.mkdir(parents=True, exist_ok=True)

        obj.to_csv(path, index=False)

        context.log.info(
            "Wrote %s rows to %s",
            len(obj),
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
        """Load a persisted CSV asset as a DataFrame."""
        path = self._get_path(context)

        context.log.info("Loading asset from %s", path)

        return pd.read_csv(path)
