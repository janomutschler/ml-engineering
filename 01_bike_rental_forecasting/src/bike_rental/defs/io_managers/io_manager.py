"""LakeFS-backed Parquet IO manager for persisting and versioning DataFrame assets."""

import pandas as pd
from dagster import ConfigurableIOManager, InputContext, OutputContext

from bike_rental.defs.resources.lakefs import LakeFSResource


class LakeFSParquetIOManager(ConfigurableIOManager):
    """Persist DataFrame assets as Parquet on a per-run LakeFS branch.

    Owns its own LakeFS connection rather than injecting it as a resource
    dependency, which avoids Dagster's nested-resource injection quirks for
    IO managers.
    """

    host: str
    access_key: str
    secret_key: str
    repository: str = "bike-rental"
    prefix: str = "processed"

    def _lakefs(self) -> LakeFSResource:
        """Build a LakeFSResource from this IO manager's connection fields."""
        return LakeFSResource(
            host=self.host,
            access_key=self.access_key,
            secret_key=self.secret_key,
            repository=self.repository,
        )

    def _object_path(self, context: OutputContext | InputContext) -> str:
        asset_name = context.asset_key.path[-1]
        return f"{self.prefix}/{asset_name}.parquet"

    def handle_output(self, context: OutputContext, obj: pd.DataFrame) -> None:
        """Write a DataFrame asset as Parquet to the run branch."""
        lfs = self._lakefs()
        run_id = context.step_context.run_id
        branch = lfs.run_branch(run_id)
        lfs.ensure_branch(branch)

        uri = lfs.object_uri(branch, self._object_path(context))
        obj.to_parquet(uri, index=False, storage_options=lfs.storage_options())

        context.log.info(
            "Wrote asset '%s' (%s rows, %s columns) to %s",
            context.asset_key.to_user_string(),
            len(obj),
            len(obj.columns),
            uri,
        )
        context.add_output_metadata(
            {
                "lakefs_uri": uri,
                "rows": len(obj),
                "columns": len(obj.columns),
            }
        )

    def load_input(self, context: InputContext) -> pd.DataFrame:
        """Load a Parquet asset from the run branch, or main outside a run.

        The run branch is created from ``main`` on demand. Branch creation in
        LakeFS is zero-copy, so a fresh branch is an instant snapshot of the
        last published data. Within a run this means reads see anything written
        earlier in the same run plus the published version of everything else;
        in a partial rematerialization the branch is pure snapshot-of-main, so
        training alone always reads the last published data.
        """
        lfs = self._lakefs()

        try:
            run_id = context.step_context.run_id
        except Exception:
            run_id = None

        if run_id:
            branch = lfs.run_branch(run_id)
            lfs.ensure_branch(branch)
        else:
            branch = "main"

        uri = lfs.object_uri(branch, self._object_path(context))
        context.log.info(
            "Loading upstream asset '%s' from %s",
            context.asset_key.to_user_string(),
            uri,
        )
        return pd.read_parquet(uri, storage_options=lfs.storage_options())
