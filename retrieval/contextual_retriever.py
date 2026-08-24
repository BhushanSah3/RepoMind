"""Dependency-graph context enrichment for already-ranked retrieval results."""

from collections.abc import Iterable

from ingestion.ast_chunker import CodeChunk
from ingestion.dependency_graph import DependencyGraph
from retrieval.hybrid_retriever import RetrievalResult


class ContextualRetriever:
    """Append related module summaries after primary results without changing their ranking."""

    def __init__(
        self,
        dependency_graph: DependencyGraph,
        chunks: Iterable[CodeChunk],
        max_context_chunks: int = 4,
    ) -> None:
        self.dependency_graph = dependency_graph
        self.max_context_chunks = max(0, max_context_chunks)
        self._module_chunks = {
            chunk.file_path: chunk for chunk in chunks if chunk.chunk_type == "module"
        }

    def enrich(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Append up to four related-file module chunks, marked as supplemental context."""

        if not results or self.max_context_chunks == 0:
            return results

        existing_ids = {result.chunk_id for result in results}
        context: list[RetrievalResult] = []
        for result in results:
            file_path = result.metadata.get("file_path")
            if not isinstance(file_path, str):
                continue
            for related_file in self.dependency_graph.get_related_files(file_path, depth=1):
                chunk = self._module_chunks.get(related_file)
                if chunk is None or chunk.chunk_id in existing_ids:
                    continue
                existing_ids.add(chunk.chunk_id)
                metadata = chunk.model_dump(exclude={"content"})
                metadata["is_context"] = True
                context.append(
                    RetrievalResult(
                        chunk_id=chunk.chunk_id,
                        content=chunk.content,
                        metadata=metadata,
                        rrf_score=0.0,
                    )
                )
                if len(context) >= self.max_context_chunks:
                    return [*results, *context]
        return [*results, *context]


def enrich_with_context(
    results: list[RetrievalResult],
    dependency_graph: DependencyGraph,
    chunks: Iterable[CodeChunk],
    max_context_chunks: int = 4,
) -> list[RetrievalResult]:
    """Convenience API for dependency-aware context enrichment."""

    return ContextualRetriever(dependency_graph, chunks, max_context_chunks).enrich(results)
