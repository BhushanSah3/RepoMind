"""Security and bug-analysis specialist."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from agents.architect_agent import _generate
from agents.state import AgentState, trace_event
from llm.provider import LLMProvider


BUG_HUNTER_SYSTEM_PROMPT = """You are a senior security engineer and bug hunter. Analyze only supplied code.
For each substantiated issue provide severity, file/line location, evidence, impact, and a concrete fix.
Check injection, secrets, validation, authorization, error handling, races, deserialization, and resource leaks.
Use the dependency/call-graph context to understand data flow across files.
Do not fabricate findings; say when evidence is insufficient."""


class BugHunter:
    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        dependency_graph: Any | None = None,
    ) -> None:
        self.llm_provider = llm_provider or LLMProvider()
        self.dependency_graph = dependency_graph

    def node(self, state: AgentState) -> dict[str, Any]:
        started_at = perf_counter()
        # Include call-graph context so the LLM can trace data flow across files
        graph_context = ""
        if self.dependency_graph is not None:
            try:
                graph_context = (
                    f"\n\nCall/import graph context:\n"
                    f"{self.dependency_graph.get_architecture_summary()}"
                )
            except Exception:
                pass
        output = _generate(
            self.llm_provider,
            BUG_HUNTER_SYSTEM_PROMPT,
            state["query"],
            state.get("retrieved_chunks", []),
            graph_context,
        )
        return {
            "agent_outputs": {"bug_hunter": output},
            "agent_trace": [trace_event("bug_hunter", "reviewed retrieved code for defects", started_at)],
        }
