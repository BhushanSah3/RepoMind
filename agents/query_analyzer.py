"""Classify a repository question and route it to focused specialists."""

from __future__ import annotations

import re
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from agents.state import AgentState, trace_event
from llm.provider import LLMProvider


QUERY_ANALYZER_SYSTEM_PROMPT = """You are RepoMind's query analysis agent. Classify the question and select one or
two specialists: code_retriever (find code/configuration), architect (structure,
dependencies, design), bug_hunter (bugs/security), code_reviewer (improvements).
Use general only for greetings or non-code conversation. Return structured output."""

AgentName = Literal["code_retriever", "architect", "bug_hunter", "code_reviewer"]


class QueryAnalysis(BaseModel):
    intent: Literal["code_lookup", "architecture", "bug_security", "code_review", "general"]
    scope: str = Field(description="Target code area, or full_repo when unclear")
    complexity: Literal["simple", "moderate", "complex"]
    target_agents: list[AgentName] = Field(default_factory=lambda: ["code_retriever"], min_length=1, max_length=2)
    reasoning: str


class QueryAnalyzer:
    """LLM-first query analysis with a deterministic, offline-safe fallback."""

    def __init__(self, llm_provider: LLMProvider | None = None) -> None:
        self.llm_provider = llm_provider or LLMProvider()

    def analyze(self, query: str) -> QueryAnalysis:
        fallback = _heuristic_analysis(query)
        try:
            model = self.llm_provider.get_chat_model("classification")
            structured = model.with_structured_output(QueryAnalysis)
            result = structured.invoke([
                ("system", QUERY_ANALYZER_SYSTEM_PROMPT),
                ("human", query),
            ])
            return result if isinstance(result, QueryAnalysis) else QueryAnalysis.model_validate(result)
        except Exception:
            return fallback

    def node(self, state: AgentState) -> dict[str, Any]:
        started_at = perf_counter()
        analysis = self.analyze(state["query"])
        return {
            "intent": analysis.intent,
            "scope": analysis.scope,
            "complexity": analysis.complexity,
            "target_agents": analysis.target_agents,
            "agent_trace": [trace_event("query_analyzer", f"classified as {analysis.intent}", started_at, reasoning=analysis.reasoning)],
        }


def _heuristic_analysis(query: str) -> QueryAnalysis:
    normalized = query.lower()
    scope = _extract_scope(query)
    if re.fullmatch(r"\s*(hi|hello|hey|thanks?|thank you)[!.\s]*", normalized):
        return QueryAnalysis(intent="general", scope="full_repo", complexity="simple", target_agents=["code_retriever"], reasoning="Conversational query.")

    security_terms = ("security", "vulnerab", "injection", "xss", "csrf", "secret", "exploit", "unsafe", "bug", "race condition")
    review_terms = ("review", "improve", "refactor", "best practice", "code quality", "clean up")
    architecture_terms = ("architecture", "structure", "organized", "design pattern", "data flow", "dependency", "how does", "how do")
    complex_terms = ("flow", "across", "end-to-end", "all", "security", "architecture", "dependency")

    if any(term in normalized for term in security_terms):
        targets: list[AgentName] = ["bug_hunter"]
        if any(term in normalized for term in ("flow", "module", "across", "auth")):
            targets.append("architect")
        intent = "bug_security"
    elif any(term in normalized for term in review_terms):
        targets = ["code_reviewer"]
        intent = "code_review"
    elif any(term in normalized for term in architecture_terms):
        targets = ["architect"]
        intent = "architecture"
    else:
        targets = ["code_retriever"]
        intent = "code_lookup"

    complexity: Literal["simple", "moderate", "complex"] = "complex" if len(targets) == 2 or any(term in normalized for term in complex_terms) else "moderate"
    return QueryAnalysis(intent=intent, scope=scope, complexity=complexity, target_agents=targets, reasoning="Deterministic keyword routing fallback.")


def _extract_scope(query: str) -> str:
    match = re.search(r"(?:in|for|of|about|review)\s+(?:the\s+)?([\w./-]+(?:\s+(?:module|file|class|function|layer))?)", query, flags=re.IGNORECASE)
    return match.group(1).strip(" ?.") if match else "full_repo"
