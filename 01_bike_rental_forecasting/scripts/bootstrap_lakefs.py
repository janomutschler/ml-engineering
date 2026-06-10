"""Create the LakeFS repository for the project (idempotent).

Run once after starting LakeFS:

    make lakefs-repo

Requires LAKEFS_HOST, LAKEFS_ACCESS_KEY, and LAKEFS_SECRET_KEY in the env.
"""

import os

import lakefs
from lakefs.client import Client


def main() -> None:
    """Create the configured repository if it does not already exist."""
    host = os.environ["LAKEFS_HOST"]
    access_key = os.environ["LAKEFS_ACCESS_KEY"]
    secret_key = os.environ["LAKEFS_SECRET_KEY"]
    repository = os.environ.get("LAKEFS_REPOSITORY", "bike-rental")

    client = Client(host=host, username=access_key, password=secret_key)

    try:
        lakefs.Repository(repository, client=client).create(
            storage_namespace=f"local://{repository}",
            exist_ok=True,
        )
        print(f"Repository ready: {repository}")
    except lakefs.exceptions.LakeFSException as error:
        print(f"Failed to create repository {repository!r}: {error}")
        raise


if __name__ == "__main__":
    main()