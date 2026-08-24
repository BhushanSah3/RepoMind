"""Best-effort local dependency graph construction and persistence."""

import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from exceptions import IngestionError
from ingestion.ast_chunker import CodeChunk


class DependencyGraph:
    """Directed graph of local files, classes, functions, and their relationships."""

    def __init__(self) -> None:
        try:
            import networkx as nx
        except ImportError as error:
            raise IngestionError("Dependency graph support is not installed. Install networkx.") from error
        self._nx = nx
        self.graph = nx.DiGraph()

    def build_from_chunks(self, chunks: list[CodeChunk]) -> None:
        """Build graph nodes and best-effort import, call, and inheritance edges."""

        self.graph.clear()
        file_chunks: dict[str, list[CodeChunk]] = defaultdict(list)
        for chunk in chunks:
            file_chunks[chunk.file_path].append(chunk)

        known_files = set(file_chunks)
        functions_by_name: dict[str, list[str]] = defaultdict(list)
        classes_by_name: dict[str, list[str]] = defaultdict(list)
        for file_path, group in file_chunks.items():
            language = group[0].language
            self.graph.add_node(file_path, node_type="file", path=file_path, language=language, chunk_count=len(group))
            for chunk in group:
                if chunk.chunk_type == "class":
                    self.graph.add_node(chunk.chunk_id, node_type="class", file_path=file_path, methods=[])
                    self.graph.add_edge(file_path, chunk.chunk_id, relation="contains")
                    classes_by_name[chunk.name].append(chunk.chunk_id)
                elif chunk.chunk_type == "function":
                    self.graph.add_node(chunk.chunk_id, node_type="function", file_path=file_path, params=chunk.params)
                    self.graph.add_edge(file_path, chunk.chunk_id, relation="contains")
                    if chunk.parent_class:
                        class_id = f"{file_path}::class::{chunk.parent_class}"
                        if class_id in self.graph:
                            self.graph.add_edge(class_id, chunk.chunk_id, relation="contains")
                            self.graph.nodes[class_id]["methods"].append(chunk.name)
                    functions_by_name[chunk.name].append(chunk.chunk_id)

        for file_path, group in file_chunks.items():
            for chunk in group:
                if chunk.chunk_type == "module" and chunk.language == "python":
                    for target in _resolve_imports(file_path, chunk.imports, known_files):
                        self.graph.add_edge(file_path, target, relation="imports")
                if chunk.chunk_type == "class" and chunk.language == "python":
                    self._add_inheritance_edges(chunk, classes_by_name)
                if chunk.chunk_type == "function" and chunk.language == "python":
                    self._add_call_edges(chunk, functions_by_name)

    def get_related_files(self, file_path: str, depth: int = 2) -> list[str]:
        """Return file nodes within the requested dependency-hop distance."""

        if file_path not in self.graph:
            return []
        related = self._nx.ego_graph(self.graph.to_undirected(), file_path, radius=depth)
        return sorted(node for node, data in related.nodes(data=True) if data.get("node_type") == "file" and node != file_path)

    def get_architecture_summary(self) -> str:
        """Summarize file structure, dependency hubs, entry points, and cycles."""

        files = [node for node, data in self.graph.nodes(data=True) if data.get("node_type") == "file"]
        import_graph = self.graph.edge_subgraph(
            (source, target) for source, target, data in self.graph.edges(data=True) if data.get("relation") == "imports"
        )
        hubs = sorted(files, key=lambda item: self.graph.degree(item), reverse=True)[:5]
        entry_points = sorted(node for node in files if import_graph.in_degree(node) == 0)
        leaves = sorted(node for node in files if import_graph.out_degree(node) == 0)
        cycles = list(self._nx.simple_cycles(import_graph))
        return "\n".join(
            [
                f"Files: {len(files)}",
                f"Most connected: {', '.join(hubs) or 'none'}",
                f"Entry points: {', '.join(entry_points) or 'none'}",
                f"Leaf modules: {', '.join(leaves) or 'none'}",
                f"Circular dependencies: {len(cycles)}",
            ]
        )

    def serialize(self) -> dict[str, Any]:
        """Return JSON-compatible graph data for persistent storage."""

        return self._nx.node_link_data(self.graph)

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "DependencyGraph":
        """Reconstruct a graph previously returned by :meth:`serialize`."""

        instance = cls()
        instance.graph = instance._nx.node_link_graph(data, directed=True)
        return instance

    def save(self, path: Path) -> None:
        """Persist the graph as JSON alongside repository index data."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.serialize()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DependencyGraph":
        """Load a persisted graph."""

        try:
            return cls.deserialize(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            raise IngestionError(f"Could not load dependency graph: {path}") from error

    def _add_call_edges(self, chunk: CodeChunk, functions_by_name: dict[str, list[str]]) -> None:
        try:
            tree = ast.parse(chunk.content)
        except SyntaxError:
            return
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            candidates = functions_by_name.get(node.func.id, [])
            if len(candidates) == 1 and candidates[0] != chunk.chunk_id:
                self.graph.add_edge(chunk.chunk_id, candidates[0], relation="calls")

    def _add_inheritance_edges(self, chunk: CodeChunk, classes_by_name: dict[str, list[str]]) -> None:
        try:
            node = next(item for item in ast.parse(chunk.content).body if isinstance(item, ast.ClassDef))
        except (SyntaxError, StopIteration):
            return
        for base in node.bases:
            base_name = base.id if isinstance(base, ast.Name) else None
            candidates = classes_by_name.get(base_name or "", [])
            if len(candidates) == 1 and candidates[0] != chunk.chunk_id:
                self.graph.add_edge(chunk.chunk_id, candidates[0], relation="inherits")


def _resolve_imports(file_path: str, imports: list[str], known_files: set[str]) -> set[str]:
    targets: set[str] = set()
    for statement in imports:
        try:
            node = ast.parse(statement).body[0]
        except SyntaxError:
            continue
        module = node.module if isinstance(node, ast.ImportFrom) else None
        names = [alias.name for alias in node.names] if isinstance(node, (ast.Import, ast.ImportFrom)) else []
        candidates = [module] if module else names
        for candidate in filter(None, candidates):
            base = Path(file_path).parent
            if isinstance(node, ast.ImportFrom) and node.level:
                for _ in range(node.level - 1):
                    base = base.parent
                candidate_path = base / candidate.replace(".", "/")
            else:
                candidate_path = Path(candidate.replace(".", "/"))
            for path in (candidate_path.with_suffix(".py"), candidate_path / "__init__.py"):
                normalized = path.as_posix()
                if normalized in known_files:
                    targets.add(normalized)
    return targets
