"""Git helpers for capturing run provenance."""

import subprocess


def get_git_commit() -> str | None:
    """Return the short git commit SHA of the working tree.

    Returns
    -------
    str | None
        The short commit SHA, or None if git is unavailable or the working
        directory is not a repository.

    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    return result.stdout.strip()
