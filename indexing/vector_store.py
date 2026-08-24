"""Persistent ChromaDB storage for RepoMind code chunks."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from exceptions import RetrievalError
from indexing.embedder import LocalEmbedder
from ingestion.ast_chunker import CodeChunk


@dataclass(frozen=True)
class VectorSearchResult:
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    score: float


class ChromaStore:
    """A persistent, isolated Chroma collection for one analyzed repository."""

    def __init__(self, repo_slug: str, persist_directory: Path, embedder: LocalEmbedder) -> None:
        self.collection_name = sanitize_collection_name(repo_slug)
        self.persist_directory = persist_directory
        self.embedder = embedder
        self._collection: Any | None = None

    def add_documents(self, chunks: list[CodeChunk]) -> None:
        """Embed and upsert chunks, retaining all retrieval-relevant metadata."""

        if not chunks:
            return
        collection = self._get_collection()
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            embeddings=self.embedder.embed_texts([chunk.content for chunk in chunks]),
            metadatas=[_metadata(chunk) for chunk in chunks],
        )

    def search(
        self, query_embedding: list[float], top_k: int = 10, where: dict[str, Any] | None = None
    ) -> list[VectorSearchResult]:
        """Return dense-search matches ordered by Chroma cosine distance."""

        result = self._get_collection().query(
            query_embeddings=[query_embedding], n_results=top_k, where=where, include=["documents", "metadatas", "distances"]
        )
        return [
            VectorSearchResult(chunk_id, content, metadata, 1 - distance)
            for chunk_id, content, metadata, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def delete_collection(self) -> None:
        """Remove this repository's vector index before re-ingestion."""

        try:
            self._get_client().delete_collection(self.collection_name)
        except Exception as error:
            # ChromaDB versions use different exception classes for a missing
            # collection (ValueError, NotFoundError, or InvalidCollection).
            # A missing collection is the normal first-ingestion state.
            if "does not exist" not in str(error).lower() and "not found" not in str(error).lower():
                raise
        self._collection = None

    def get_collection_stats(self) -> dict[str, int]:
        collection = self._get_collection()
        return {"chunk_count": collection.count()}

    def _get_collection(self) -> Any:
        if self._collection is None:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._collection = self._get_client().get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def _get_client(self) -> Any:
        try:
            import chromadb
        except ImportError as error:
            raise RetrievalError("Vector-store support is not installed. Install chromadb.") from error
        return chromadb.PersistentClient(path=str(self.persist_directory))


def sanitize_collection_name(repo_slug: str) -> str:
    """Return a Chroma-compatible, stable collection name."""

    name = re.sub(r"[^a-zA-Z0-9_-]", "_", repo_slug).strip("_-").lower()
    return (name or "repository")[:63]


def _metadata(chunk: CodeChunk) -> dict[str, Any]:
    data = chunk.model_dump(exclude={"content", "chunk_id"}, exclude_none=True)
    return {key: json.dumps(value) if isinstance(value, list) else value for key, value in data.items()}
