"""Shared state and small utilities for the RepoMind agent graph."""

from __future__ import annotations

import operator
from time import perf_counter
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
else:
    BaseMessage = Any


Intent = Literal["code_lookup", "architecture", "bug_security", "code_review", "general"]
Complexity = Literal["simple", "moderate", "complex"]


class AgentState(TypedDict, total=False):
    """Values shared between nodes in the RepoMind workflow.

    ``total=False`` lets callers submit just ``query`` at the graph boundary.
    LangGraph applies the reducers when it is installed; importing this module
    deliberately does not require LangGraph, which keeps non-UI tools usable.
    """

    query: str
    chat_history: Annotated[list[BaseMessage], operator.add]
    intent: Intent
    scope: str
    complexity: Complexity
    target_agents: list[str]
    retrieved_chunks: list[dict[str, Any]]
    retrieval_queries: list[str]
    retrieval_strategy: str
    agent_outputs: dict[str, str]
    synthesized_answer: str
    faithfulness_score: float
    context_relevance_score: float
    answer_relevance_score: float
    evaluation_passed: bool
    retry_count: int
    agent_trace: Annotated[list[dict[str, Any]], operator.add]


def trace_event(agent: str, action: str, started_at: float | None = None, **details: Any) -> dict[str, Any]:
    """Build one UI-safe agent trace record."""

    event: dict[str, Any] = {"agent": agent, "action": action}
    if started_at is not None:
        event["duration_ms"] = round((perf_counter() - started_at) * 1000)
    event.update(details)
    return event
