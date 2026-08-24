"""Deterministic repository file discovery with code-focused filtering."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from config.settings import get_settings
from exceptions import IngestionError


logger = logging.getLogger(__name__)

SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    "venv",
    ".venv",
    "env",
    ".env",
    "dist",
    "build",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "egg-info",
    ".eggs",
    "vendor",
    "third_party",
}
SKIP_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".so",
    ".dll",
    ".exe",
    ".bin",
    ".dat",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".woff",
    ".ttf",
    ".eot",
    ".mp3",
    ".mp4",
    ".zip",
    ".tar",
    ".gz",
}
LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".md": "markdown",
    ".rst": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".cfg": "config",
    ".ini": "config",
    ".txt": "text",
}


@dataclass(frozen=True)
class FileInfo:
    """Metadata for one source file selected for RepoMind ingestion."""

    path: Path
    relative_path: str
    extension: str
    size_bytes: int
    language: str


def walk_repository(repo_path: Path, max_file_size_bytes: int | None = None) -> list[FileInfo]:
    """Return supported repository files in a stable processing order."""

    if not repo_path.is_dir():
        raise IngestionError(f"Repository path does not exist: {repo_path}")

    size_limit = max_file_size_bytes or get_settings().max_file_size_bytes
    files: list[FileInfo] = []
    for directory, directory_names, file_names in os.walk(repo_path):
        directory_names[:] = [
            name
            for name in directory_names
            if not _should_skip_directory(name)
        ]

        current_directory = Path(directory)
        for file_name in file_names:
            path = current_directory / file_name
            extension = path.suffix.lower()
            # Runtime environment files may contain API keys and must never be
            # sent to embeddings, keyword search, or an LLM prompt.
            if file_name == ".env" or (file_name.startswith(".env.") and file_name != ".env.example"):
                continue
            if extension in SKIP_EXTENSIONS:
                continue
            try:
                size_bytes = path.stat().st_size
            except OSError:
                logger.debug("Skipped unreadable file: %s", path)
                continue
            if size_bytes > size_limit:
                logger.debug("Skipped oversized file: %s", path)
                continue

            files.append(
                FileInfo(
                    path=path,
                    relative_path=path.relative_to(repo_path).as_posix(),
                    extension=extension,
                    size_bytes=size_bytes,
                    language=LANGUAGE_MAP.get(extension, "text"),
                )
            )

    return sorted(files, key=lambda item: (item.language, item.relative_path))


def _should_skip_directory(name: str) -> bool:
    return (
        name in SKIP_DIRS
        or name.endswith(".egg-info")
        or (name.startswith(".") and name != ".github")
    )
