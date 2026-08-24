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
Do not fabricate findings; say when evidence is insufficient."""


class BugHunter:
    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        self.llm_provider = llm_provider or LLMProvider()

    def node(self, state: AgentState) -> dict[str, Any]:
        started_at = perf_counter()
        output = _generate(self.llm_provider, BUG_HUNTER_SYSTEM_PROMPT, state["query"], state.get("retrieved_chunks", []))
        return {
            "agent_outputs": {"bug_hunter": output},
            "agent_trace": [trace_event("bug_hunter", "reviewed retrieved code for defects", started_at)],
        }
