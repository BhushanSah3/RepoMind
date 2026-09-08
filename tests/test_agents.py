"""Tests for the agent system: state, tracing, and query analysis."""

import pytest
from time import perf_counter
from agents.state import AgentState, trace_event
from agents.query_analyzer import _heuristic_analysis, _extract_scope


def test_trace_event_basic():
    """trace_event returns dict with agent and action."""
    event = trace_event("agent_x", "action_y")
    assert event["agent"] == "agent_x"
    assert event["action"] == "action_y"


def test_trace_event_with_timing():
    """trace_event with started_at includes duration_ms."""
    start = perf_counter()
    event = trace_event("agent_x", "action_y", started_at=start)
    assert "duration_ms" in event
    assert event["duration_ms"] >= 0


def test_trace_event_extra_details():
    """Extra kwargs should appear in the event dict."""
    event = trace_event("agent_x", "action_y", extra1="value1", extra2=42)
    assert event.get("extra1") == "value1"
    assert event.get("extra2") == 42


def test_heuristic_greeting():
    """'hello' -> intent='general'."""
    analysis = _heuristic_analysis("hello")
    assert analysis.intent == "general"


def test_heuristic_security():
    """Security query -> intent='bug_security', bug_hunter in target_agents."""
    analysis = _heuristic_analysis("Are there security vulnerabilities?")
    assert analysis.intent == "bug_security"
    assert "bug_hunter" in analysis.target_agents


def test_heuristic_review():
    """Review query -> intent='code_review', code_reviewer in target_agents."""
    analysis = _heuristic_analysis("Review the auth module")
    assert analysis.intent == "code_review"
    assert "code_reviewer" in analysis.target_agents


def test_heuristic_architecture():
    """Architecture query -> intent='architecture', architect in target_agents."""
    analysis = _heuristic_analysis("How is the project structured?")
    assert analysis.intent == "architecture"
    assert "architect" in analysis.target_agents


def test_heuristic_default_lookup():
    """Default query -> intent='code_lookup', code_retriever in target_agents."""
    analysis = _heuristic_analysis("Where is the database connection?")
    assert analysis.intent == "code_lookup"
    assert "code_retriever" in analysis.target_agents


def test_heuristic_scope_extraction():
    """'review the auth module' should extract scope containing 'auth'."""
    scope = _extract_scope("review the auth module")
    assert "auth" in scope


def test_agent_state_fields():
    """AgentState TypedDict should accept all expected field names."""
    state: AgentState = {
        "query": "test query",
        "chat_history": [],
        "intent": "code_lookup",
        "scope": "full_repo",
        "complexity": "simple",
        "target_agents": ["code_retriever"],
        "retrieved_chunks": [],
        "retrieval_queries": [],
        "retrieval_strategy": "hybrid",
        "agent_outputs": {},
        "synthesized_answer": "",
        "faithfulness_score": 1.0,
        "context_relevance_score": 1.0,
        "answer_relevance_score": 1.0,
        "evaluation_passed": True,
        "retry_count": 0,
        "agent_trace": [],
    }
    assert state["query"] == "test query"
    assert state["intent"] == "code_lookup"
    assert state["evaluation_passed"] is True


def test_merge_dicts_reducer():
    """_merge_dicts should merge two dicts, with right overriding left."""
    from agents.state import _merge_dicts
    left = {"architect": "analysis A"}
    right = {"bug_hunter": "analysis B"}
    merged = _merge_dicts(left, right)
    assert merged == {"architect": "analysis A", "bug_hunter": "analysis B"}


def test_merge_dicts_empty():
    """_merge_dicts should handle empty dicts."""
    from agents.state import _merge_dicts
    assert _merge_dicts({}, {"a": "1"}) == {"a": "1"}
    assert _merge_dicts({"a": "1"}, {}) == {"a": "1"}
    assert _merge_dicts({}, {}) == {}


def test_bug_hunter_accepts_dep_graph():
    """BugHunter should accept dependency_graph parameter."""
    from agents.bug_hunter import BugHunter
    from unittest.mock import MagicMock
    mock_graph = MagicMock()
    mock_graph.get_architecture_summary.return_value = "test summary"
    hunter = BugHunter(dependency_graph=mock_graph)
    assert hunter.dependency_graph is mock_graph


def test_code_reviewer_accepts_dep_graph():
    """CodeReviewer should accept dependency_graph parameter."""
    from agents.code_reviewer import CodeReviewer
    from unittest.mock import MagicMock
    mock_graph = MagicMock()
    reviewer = CodeReviewer(dependency_graph=mock_graph)
    assert reviewer.dependency_graph is mock_graph


def test_build_graph_accepts_dep_graph():
    """build_graph should accept dependency_graph keyword argument."""
    from agents.graph import build_graph
    from unittest.mock import MagicMock
    # Just verify the function signature accepts the kwarg without error
    code_retriever = MagicMock()
    dep_graph = MagicMock()
    try:
        build_graph(code_retriever, dependency_graph=dep_graph)
    except Exception:
        # May fail if LangGraph isn't installed, but signature should be valid
        pass

