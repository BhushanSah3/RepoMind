"""Safe, shallow cloning of public GitHub repositories."""

import logging
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from exceptions import IngestionError


logger = logging.getLogger(__name__)

GITHUB_HOST = "github.com"
LARGE_REPOSITORY_BYTES = 500 * 1024 * 1024


def clone_repo(github_url: str) -> Path:
    """Clone a GitHub repository to an isolated temporary directory.

    The clone is shallow because RepoMind analyzes the working tree rather than
    commit history. Call :func:`cleanup` when analysis is finished.
    """

    normalized_url = _validate_github_url(github_url)
    target_dir = Path(tempfile.mkdtemp(prefix="repomind_"))

    try:
        from git import GitCommandError, Repo
    except ImportError as error:
        cleanup(target_dir)
        raise IngestionError("Git support is not installed. Install GitPython to clone repositories.") from error

    try:
        Repo.clone_from(normalized_url, target_dir, depth=1)
    except GitCommandError as error:
        cleanup(target_dir)
        raise IngestionError(
            "Could not clone the repository. It may be private, unavailable, or invalid."
        ) from error

    repository_size = _directory_size(target_dir)
    if repository_size > LARGE_REPOSITORY_BYTES:
        logger.warning(
            "Repository %s is %.1f MB; large repositories may take longer to process.",
            normalized_url,
            repository_size / 1024 / 1024,
        )
    return target_dir


def cleanup(repo_path: Path) -> None:
    """Delete a RepoMind-owned temporary clone without touching other paths."""

    if not repo_path.exists():
        return

    resolved_path = repo_path.resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    if not resolved_path.is_relative_to(temp_root) or not resolved_path.name.startswith("repomind_"):
        raise IngestionError("Refusing to delete a path that is not a RepoMind temporary clone.")
    shutil.rmtree(resolved_path)


def _validate_github_url(github_url: str) -> str:
    """Return a normalized HTTPS GitHub repository URL or raise a clear error."""

    candidate = github_url.strip()
    parsed = urlparse(candidate)
    parts = [part for part in parsed.path.split("/") if part]
    if (
        parsed.scheme != "https"
        or parsed.hostname != GITHUB_HOST
        or len(parts) != 2
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Invalid GitHub URL. Expected format: https://github.com/owner/repository")

    owner, repository = parts
    if repository.endswith(".git"):
        repository = repository[:-4]
    if not owner or not repository:
        raise ValueError("Invalid GitHub URL. Expected format: https://github.com/owner/repository")
    return f"https://{GITHUB_HOST}/{owner}/{repository}.git"


def _directory_size(directory: Path) -> int:
    """Calculate directory size while tolerating files that disappear mid-scan."""

    total_size = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file():
                total_size += path.stat().st_size
        except OSError:
            logger.debug("Skipped unreadable path while sizing clone: %s", path)
    return total_size
