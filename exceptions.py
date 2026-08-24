"""Domain exceptions with messages that are safe to show in the UI."""


class RepoMindError(Exception):
    """Base exception for recoverable RepoMind errors."""


class IngestionError(RepoMindError):
    """Raised when a repository cannot be ingested."""


class RetrievalError(RepoMindError):
    """Raised when code retrieval cannot be completed."""


class LLMError(RepoMindError):
    """Raised when no configured language-model provider is available."""
