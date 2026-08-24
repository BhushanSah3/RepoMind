"""Code-aware BM25 keyword index with local persistence."""

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from exceptions import RetrievalError
from ingestion.ast_chunker import CodeChunk


_CAMEL_CASE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_TOKENS = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|==|!=|>=|<=|\S")


@dataclass(frozen=True)
class BM25Result:
    chunk: CodeChunk
    score: float


class BM25Index:
    """Sparse search that preserves code identifiers and operators."""

    def __init__(self) -> None:
        self._index = None
        self._chunks: list[CodeChunk] = []

    def build_index(self, chunks: list[CodeChunk]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as error:
            raise RetrievalError("BM25 support is not installed. Install rank-bm25.") from error
        self._chunks = chunks
        self._index = BM25Okapi([tokenize_code(chunk.content) for chunk in chunks])

    def search(self, query: str, top_k: int = 10) -> list[BM25Result]:
        if self._index is None:
            raise RetrievalError("BM25 index has not been built.")
        scores = self._index.get_scores(tokenize_code(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [BM25Result(self._chunks[index], float(score)) for index, score in ranked]

    def save(self, path: Path) -> None:
        if self._index is None:
            raise RetrievalError("BM25 index has not been built.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump((self._index, self._chunks), handle)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        try:
            with path.open("rb") as handle:
                index, chunks = pickle.load(handle)
        except (OSError, pickle.UnpicklingError) as error:
            raise RetrievalError(f"Could not load BM25 index: {path}") from error
        instance = cls()
        instance._index, instance._chunks = index, chunks
        return instance


def tokenize_code(text: str) -> list[str]:
    """Tokenize identifiers, camelCase, snake_case, operators, and literals."""

    tokens: list[str] = []
    for token in _TOKENS.findall(text):
        expanded = _CAMEL_CASE.sub(" ", token).replace("_", " ").split()
        tokens.extend(part.lower() for part in expanded if len(part) > 1 or part in {"i", "x", "y"})
    return tokens
