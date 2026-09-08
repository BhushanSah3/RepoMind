"""Dependency-graph context enrichment for already-ranked retrieval results."""

import re
from collections.abc import Iterable
from pathlib import Path

from ingestion.ast_chunker import CodeChunk
from ingestion.dependency_graph import DependencyGraph
from retrieval.hybrid_retriever import RetrievalResult


class ContextualRetriever:
    """Append related module, importer, and test file summaries after primary results without changing their ranking."""

    def __init__(
        self,
        dependency_graph: DependencyGraph,
        chunks: Iterable[CodeChunk],
        max_context_chunks: int = 6,
    ) -> None:
        self.dependency_graph = dependency_graph
        self.max_context_chunks = max(0, max_context_chunks)
        self._module_chunks = {
            chunk.file_path: chunk for chunk in chunks if chunk.chunk_type == "module"
        }

    def enrich(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Append related modules, importer files, and test files as supplemental context."""

        if not results or self.max_context_chunks == 0:
            return results

        existing_ids = {result.chunk_id for result in results}
        context: list[RetrievalResult] = []

        # 1. Related module chunks from dependency graph
        for result in results:
            file_path = self._extract_file_path(result)
            if not file_path:
                continue
            try:
                related_files = self.dependency_graph.get_related_files(file_path, depth=1)
            except AttributeError:
                related_files = []
            for related_file in (related_files or []):
                chunk = self._get_module_chunk(related_file)
                if chunk is None or chunk.chunk_id in existing_ids:
                    continue
                existing_ids.add(chunk.chunk_id)
                context.append(self._create_context_result(chunk, "related_module"))
                if len(context) >= self.max_context_chunks:
                    return [*results, *context]

        # 2. Importer files that import from the same modules
        if len(context) < self.max_context_chunks:
            for result in results:
                file_path = self._extract_file_path(result)
                if not file_path:
                    continue
                try:
                    importers = self.dependency_graph.get_importers(file_path)
                except AttributeError:
                    importers = []
                for importer_file in (importers or []):
                    if importer_file == file_path:
                        continue
                    chunk = self._get_module_chunk(importer_file)
                    if chunk is None or chunk.chunk_id in existing_ids:
                        continue
                    existing_ids.add(chunk.chunk_id)
                    context.append(self._create_context_result(chunk, "importer"))
                    if len(context) >= self.max_context_chunks:
                        return [*results, *context]

        # 3. Corresponding test files in _module_chunks
        if len(context) < self.max_context_chunks:
            for result in results:
                file_path = self._extract_file_path(result)
                if not file_path:
                    continue
                for test_chunk in self._find_test_chunks(file_path, existing_ids):
                    if test_chunk.chunk_id in existing_ids:
                        continue
                    existing_ids.add(test_chunk.chunk_id)
                    context.append(self._create_context_result(test_chunk, "test_file"))
                    if len(context) >= self.max_context_chunks:
                        return [*results, *context]

        return [*results, *context]

    def _get_module_chunk(self, file_path: str) -> CodeChunk | None:
        """Look up a module chunk by file path with slash normalization."""
        chunk = self._module_chunks.get(file_path)
        if chunk is not None:
            return chunk
        normalized = file_path.replace("\\", "/")
        chunk = self._module_chunks.get(normalized)
        if chunk is not None:
            return chunk
        return self._module_chunks.get(file_path.replace("/", "\\"))

    def _find_test_chunks(self, file_path: str, existing_ids: set[str]) -> list[CodeChunk]:
        """Find corresponding test file chunks in _module_chunks containing 'test' in their path."""
        norm_fp = file_path.replace("\\", "/")
        stem = Path(norm_fp).stem.lower().replace("-", "_")
        if not stem:
            return []

        exact_matches: list[CodeChunk] = []
        dir_matches: list[CodeChunk] = []
        name_matches: list[CodeChunk] = []

        boundary_pattern = re.compile(rf"(^|[_/.-]){re.escape(stem)}([_/.-]|$)")

        for cand_path, chunk in self._module_chunks.items():
            if chunk.chunk_id in existing_ids:
                continue
            norm_cand = cand_path.replace("\\", "/")
            if norm_cand == norm_fp:
                continue
            if "test" not in norm_cand.lower():
                continue

            cand_stem = Path(norm_cand).stem.lower().replace("-", "_")

            # Rank 1: test_<stem> or <stem>_test (e.g., test_foo.py, foo_test.py)
            if cand_stem in {f"test_{stem}", f"{stem}_test"}:
                exact_matches.append(chunk)
            # Rank 2: Same stem inside a test directory (e.g., tests/foo.py)
            elif cand_stem == stem:
                dir_matches.append(chunk)
            # Rank 3: Bounded substring match in test filename or path
            elif boundary_pattern.search(norm_cand.lower()):
                name_matches.append(chunk)

        if exact_matches:
            return exact_matches
        if dir_matches:
            return dir_matches
        return name_matches

    @staticmethod
    def _extract_file_path(result: RetrievalResult) -> str | None:
        """Extract file path from a RetrievalResult's metadata or chunk_id."""
        metadata = getattr(result, "metadata", None)
        if isinstance(metadata, dict):
            file_path = metadata.get("file_path")
            if isinstance(file_path, str) and file_path.strip():
                return file_path
        chunk_id = getattr(result, "chunk_id", None)
        if isinstance(chunk_id, str) and "::" in chunk_id:
            candidate = chunk_id.split("::")[0].strip()
            if candidate:
                return candidate
        return None

    @staticmethod
    def _create_context_result(chunk: CodeChunk, context_type: str) -> RetrievalResult:
        """Create a RetrievalResult for an enrichment chunk with context metadata."""
        if hasattr(chunk, "model_dump"):
            metadata = chunk.model_dump(exclude={"content"})
        elif hasattr(chunk, "dict"):
            metadata = chunk.dict(exclude={"content"})
        elif hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
            metadata = dict(chunk.metadata)
        else:
            metadata = {
                "file_path": chunk.file_path,
                "chunk_type": getattr(chunk, "chunk_type", "module"),
            }
        metadata["is_context"] = True
        metadata["context_type"] = context_type
        return RetrievalResult(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            metadata=metadata,
            rrf_score=0.0,
        )


def enrich_with_context(
    results: list[RetrievalResult],
    dependency_graph: DependencyGraph,
    chunks: Iterable[CodeChunk],
    max_context_chunks: int = 6,
) -> list[RetrievalResult]:
    """Convenience API for dependency-aware context enrichment."""

    return ContextualRetriever(dependency_graph, chunks, max_context_chunks).enrich(results)
