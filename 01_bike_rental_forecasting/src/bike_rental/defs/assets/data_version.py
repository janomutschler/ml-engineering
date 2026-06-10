"""Dagster asset for committing and versioning the pipeline's data in LakeFS."""

from dagster import AssetExecutionContext, asset

from bike_rental.defs.assets.preprocessing import modeling_feature_set
from bike_rental.defs.resources.lakefs import LakeFSResource
from bike_rental.defs.utils.git import get_git_commit


@asset(deps=[modeling_feature_set])
def data_version(
    context: AssetExecutionContext,
    lakefs: LakeFSResource,
) -> str:
    """Commit the run's data branch to LakeFS and merge it to ``main``.

    Runs after all data assets have been written to the run branch. Commits the
    branch (capturing the exact state of every materialized Parquet file),
    merges it to ``main`` so the data is published, and returns the commit id.
    Downstream training logs this id to MLflow as the data lineage reference.

    Parameters
    ----------
    context : AssetExecutionContext
        Dagster execution context.
    lakefs : LakeFSResource
        LakeFS connection and versioning operations.

    Returns
    -------
    str
        The LakeFS commit id of the published data version.

    """
    branch = lakefs.run_branch(context.run_id)

    commit_id = lakefs.commit(
        branch,
        message=f"Dagster run {context.run_id}",
        metadata={
            "dagster_run_id": context.run_id,
            "git_commit": get_git_commit() or "unknown",
        },
    )
    lakefs.merge(branch, "main")

    context.log.info(
        "Committed data version %s and merged branch '%s' into main.",
        commit_id,
        branch,
    )
    context.add_output_metadata(
        {
            "lakefs_commit": commit_id,
            "branch": branch,
            "repository": lakefs.repository,
        }
    )

    return commit_id