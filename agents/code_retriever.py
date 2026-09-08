"""Agent node that turns a question into grounded, reranked code context."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Protocol

from agents.state import AgentState, trace_event


class SupportsRetrieve(Protocol):
    def retrieve(self, query: str, top_k: int = 10) -> list[Any]: ...


class CodeRetriever:
    """Run hybrid search, optional expansion, and optional local reranking.

    Dependencies are injected so this layer can be tested without downloading
    embedding/reranking models and so Streamlit can cache long-lived indexes.
    """

    def __init__(
        self,
        retriever: SupportsRetrieve,
        *,
        query_expander: Any | None = None,
        reranker: Any | None = None,
        contextual_retriever: Any | None = None,
        top_k: int = 8,
    ) -> None:
        self.retriever = retriever
        self.query_expander = query_expander
        self.reranker = reranker
        self.contextual_retriever = contextual_retriever
        self.top_k = top_k

    def retrieve(self, query: str, *, strategy: str = "hybrid") -> tuple[list[dict[str, Any]], list[str]]:
        # Adjust retrieval parameters based on retry strategy
        if strategy == "narrow":
            # Low faithfulness → fewer but more precise chunks, no expansion
            effective_top_k = max(3, self.top_k // 2)
            queries = [query]  # Skip multi-query expansion for precision
        elif strategy == "broaden":
            # Low context relevance → more chunks, more aggressive expansion
            effective_top_k = self.top_k * 2
            queries = self._expand_queries(query)
        else:
            # Default hybrid or "refocus" (refocus changes happen at synthesizer level)
            effective_top_k = self.top_k
            queries = self._expand_queries(query)

        merged: dict[str, Any] = {}
        for expanded_query in queries:
            for result in self.retriever.retrieve(expanded_query, top_k=effective_top_k):
                key = result.chunk_id
                previous = merged.get(key)
                if previous is None or getattr(result, "rrf_score", 0.0) > getattr(previous, "rrf_score", 0.0):
                    merged[key] = result

        results = sorted(merged.values(), key=lambda item: getattr(item, "rrf_score", 0.0), reverse=True)
        if self.reranker is not None and results:
            try:
                results = self.reranker.rerank(query, results, top_k=effective_top_k)
            except (ImportError, OSError, RuntimeError):
                results = results[: effective_top_k]
        else:
            results = results[: effective_top_k]

        if self.contextual_retriever is not None and results:
            try:
                results = self.contextual_retriever.enrich(results)
            except (AttributeError, OSError, RuntimeError):
                pass
        chunks = [_result_to_chunk(result) for result in results]
        return chunks, queries

    def node(self, state: AgentState) -> dict[str, Any]:
        started_at = perf_counter()
        strategy = state.get("retrieval_strategy", "hybrid")
        chunks, queries = self.retrieve(state["query"], strategy=strategy)
        return {
            "retrieved_chunks": chunks,
            "retrieval_queries": queries,
            "retrieval_strategy": strategy,
            "agent_trace": [trace_event("code_retriever", f"retrieved {len(chunks)} chunks", started_at, queries=queries)],
        }

    def _expand_queries(self, query: str) -> list[str]:
        if self.query_expander is None:
            return [query]
        try:
            variants = self.query_expander.expand(query)
        except (AttributeError, OSError, RuntimeError):
            return [query]
        return list(dict.fromkeys([query, *[item for item in variants if isinstance(item, str) and item.strip()]]))


def _result_to_chunk(result: Any) -> dict[str, Any]:
    return {
        "chunk_id": result.chunk_id,
        "content": result.content,
        "metadata": dict(getattr(result, "metadata", {}) or {}),
        "rrf_score": float(getattr(result, "rrf_score", 0.0)),
    }
