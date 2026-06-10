"""LakeFS resource for versioning the pipeline's data assets."""

import lakefs
import lakefs.exceptions
from dagster import ConfigurableResource
from lakefs.client import Client


class LakeFSResource(ConfigurableResource):
    """Connection and versioning operations for a LakeFS repository.

    LakeFS is git-for-data: nothing is versioned until an explicit commit. This
    resource owns the connection and exposes the operations the pipeline needs
    to write data to a run-scoped branch, commit it, and merge to ``main``.

    """

    host: str
    access_key: str
    secret_key: str
    repository: str = "bike-rental"

    def run_branch(self, run_id: str) -> str:
        """Return the per-run branch name for a Dagster run."""
        return f"dagster-{run_id}"

    def _client(self) -> Client:
        return Client(host=self.host, username=self.access_key, password=self.secret_key)

    def _repo(self) -> "lakefs.Repository":
        return lakefs.Repository(self.repository, client=self._client())

    def ensure_branch(self, branch: str, source: str = "main") -> None:
        """Create ``branch`` from ``source`` if it does not already exist."""
        self._repo().branch(branch).create(source_reference=source, exist_ok=True)

    def commit(self, branch: str, message: str, metadata: dict[str, str] | None = None) -> str:
        """Commit the branch and return the resulting commit id.

        If there are no uncommitted changes, returns the current branch head
        instead of failing.
        """
        target = self._repo().branch(branch)
        string_metadata = {key: str(value) for key, value in (metadata or {}).items()}
        try:
            target.commit(message=message, metadata=string_metadata)
        except lakefs.exceptions.LakeFSException:
            # Most commonly: nothing to commit. Fall back to the current head.
            pass
        return target.get_commit().id

    def merge(self, source: str, destination: str = "main") -> None:
        """Merge ``source`` branch into ``destination``.

        A 400 'no changes' response is treated as success — it means the
        branch is already identical to the destination, which is fine on
        re-runs where the data hasn't changed.
        """
        try:
            self._repo().branch(source).merge_into(destination)
        except lakefs.exceptions.BadRequestException as exc:
            if "no changes" in str(exc).lower():
                return
            raise

    def object_uri(self, branch: str, path: str) -> str:
        """Build a ``lakefs://`` URI for an object on a branch."""
        return f"lakefs://{self.repository}/{branch}/{path}"

    def storage_options(self) -> dict[str, str]:
        """Return fsspec storage options for lakefs-spec (pandas read/write)."""
        return {
            "host": self.host,
            "username": self.access_key,
            "password": self.secret_key,
        }