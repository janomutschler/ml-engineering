"""Local File-system data loading resources for the bike rental pipeline."""

from pathlib import Path

import pandas as pd
from dagster import AssetExecutionContext, ConfigurableResource

from bike_rental.defs.utils.metadata import build_dataframe_metadata


class LocalDataLoader(ConfigurableResource):
    """Load CSV source files from the local file system.

    This resource only handles file loading and optional datetime parsing.
    Schema validation and data quality checks are handled downstream in asset
    checks or transformation assets.
    """

    base_path: str = "data/sources"
    project_root: str = "."

    @property
    def resolved_base_path(self) -> Path:
        """Return the absolute path to the source data directory."""
        base_path = Path(self.base_path)

        if base_path.is_absolute():
            return base_path

        return Path(self.project_root).resolve() / base_path

    def load_csv(
        self,
        context: AssetExecutionContext,
        file_name: str,
    ) -> pd.DataFrame:
        """Load a CSV file and optionally parse datetime columns.

        Parameters
        ----------
        context : AssetExecutionContext
            Dagster context used for logging and metadata.
        file_name : str
            Name of the CSV file inside the configured base path.
        datetime_columns : Sequence[str] | None, optional
            Columns to parse as datetime values.

        Returns
        -------
        pd.DataFrame
            Loaded DataFrame.

        """
        file_path = self.resolved_base_path / file_name

        context.log.info("Loading source file: %s", file_path)

        df = pd.read_csv(file_path)

        datetime_columns = context.assets_def.metadata_by_key[context.asset_key].get(
            "datetime_columns", []
        )

        for column in datetime_columns or []:
            df[column] = pd.to_datetime(df[column], errors="raise")

        context.add_output_metadata(
            build_dataframe_metadata(
                df,
                extra_metadata={
                    "source_file": str(file_path),
                },
            )
        )

        return df
