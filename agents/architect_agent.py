"""Architecture specialist grounded in retrieved chunks and dependency data."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from agents.state import AgentState, trace_event
from llm.provider import LLMProvider
from llm.utils import extract_text


ARCHITECT_SYSTEM_PROMPT = """You are a senior software architect. Analyze only the supplied repository evidence.
Explain structure, dependencies, data flow, and patterns; cite file paths and lines.
State uncertainty rather than inventing relationships."""


class ArchitectAgent:
    def __init__(self, llm_provider: LLMProvider | None = None, dependency_graph: Any | None = None) -> None:
        self.llm_provider = llm_provider or LLMProvider()
        self.dependency_graph = dependency_graph

    def node(self, state: AgentState) -> dict[str, Any]:
        started_at = perf_counter()
        graph_context = self.dependency_graph.get_architecture_summary() if self.dependency_graph is not None else "Dependency graph unavailable."
        output = _generate(self.llm_provider, ARCHITECT_SYSTEM_PROMPT, state["query"], state.get("retrieved_chunks", []), graph_context)
        return {
            "agent_outputs": {"architect": output},
            "agent_trace": [trace_event("architect", "analyzed architecture", started_at)],
        }


def _generate(provider: LLMProvider, system: str, query: str, chunks: list[dict[str, Any]], graph_context: str = "") -> str:
    context = _format_context(chunks)
    try:
        model = provider.get_chat_model("generation")
        response = model.invoke([("system", system), ("human", f"Question: {query}\n\nDependency graph:\n{graph_context}\n\nCode evidence:\n{context}")])
        return extract_text(response)
    except Exception:
        citations = _citations(chunks)
        return "Architecture analysis requires an available LLM. Retrieved evidence: " + (", ".join(citations) or "no matching chunks.")


def _format_context(chunks: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"[{chunk.get('metadata', {}).get('file_path', chunk.get('chunk_id', 'unknown'))}]\n{chunk.get('content', '')}" for chunk in chunks)


def _citations(chunks: list[dict[str, Any]]) -> list[str]:
    return [str(chunk.get("metadata", {}).get("file_path", chunk.get("chunk_id", "unknown"))) for chunk in chunks]
