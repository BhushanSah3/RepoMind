"""Constructive code-quality review specialist."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from agents.architect_agent import _generate
from agents.state import AgentState, trace_event
from llm.provider import LLMProvider


CODE_REVIEWER_SYSTEM_PROMPT = """You are a senior code reviewer. Use only supplied code evidence.
Give constructive, actionable feedback categorized as readability, performance, maintainability, testing, or documentation.
For each suggestion cite the location, show the relevant current code, a suggested change, and reasoning.
Recognize strong patterns and avoid speculative or nitpicky feedback."""


class CodeReviewer:
    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        self.llm_provider = llm_provider or LLMProvider()

    def node(self, state: AgentState) -> dict[str, Any]:
        started_at = perf_counter()
        output = _generate(self.llm_provider, CODE_REVIEWER_SYSTEM_PROMPT, state["query"], state.get("retrieved_chunks", []))
        return {
            "agent_outputs": {"code_reviewer": output},
            "agent_trace": [trace_event("code_reviewer", "reviewed retrieved code", started_at)],
        }
