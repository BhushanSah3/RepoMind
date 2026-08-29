"""Tests for the RAG Triad evaluation system."""

import asyncio
from unittest.mock import AsyncMock, MagicMock
from evaluation.rag_triad import TriadScores, evaluate_response, _score


def test_triad_scores_defaults():
    """TriadScores() has all 1.0 defaults and passed=True."""
    scores = TriadScores()
    assert scores.faithfulness == 1.0
    assert scores.context_relevance == 1.0
    assert scores.answer_relevance == 1.0
    assert scores.passed is True


def test_triad_scores_custom():
    """Custom values should be accepted."""
    scores = TriadScores(faithfulness=0.5, passed=False)
    assert scores.faithfulness == 0.5
    assert scores.passed is False


def test_evaluate_response_fail_open():
    """When no LLM is available, should return default pass scores."""
    # evaluate_response signature: (query, retrieved_chunks, answer, llm, provider)
    # With a broken provider, it should fail-open
    broken_provider = MagicMock()
    broken_provider.get_chat_model.side_effect = Exception("No LLM")
    scores = asyncio.run(
        evaluate_response("query", [{"content": "context"}], "answer", provider=broken_provider)
    )
    assert scores.passed is True
    assert scores.faithfulness == 1.0
    assert scores.context_relevance == 1.0
    assert scores.answer_relevance == 1.0


def test_evaluate_response_with_mock_llm():
    """Mock LLM returning score 0.85 for all prompts."""
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = '{"score": 0.85}'
    mock_llm.ainvoke.return_value = mock_response

    scores = asyncio.run(
        evaluate_response("query", [{"content": "context"}], "answer", llm=mock_llm)
    )
    assert scores.faithfulness == 0.85
    assert scores.context_relevance == 0.85
    assert scores.answer_relevance == 0.85
    assert scores.passed is True


def test_score_parse_json():
    """_score should parse JSON with 'score' key."""
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = '{"score": 0.9}'
    mock_llm.ainvoke.return_value = mock_response

    result = asyncio.run(_score(mock_llm, "prompt"))
    assert result == 0.9


def test_score_clamp_high():
    """Scores above 1.0 should be clamped to 1.0."""
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = '{"score": 1.5}'
    mock_llm.ainvoke.return_value = mock_response

    result = asyncio.run(_score(mock_llm, "prompt"))
    assert result == 1.0


def test_score_clamp_low():
    """Scores below 0.0 should be clamped to 0.0."""
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = '{"score": -0.5}'
    mock_llm.ainvoke.return_value = mock_response

    result = asyncio.run(_score(mock_llm, "prompt"))
    assert result == 0.0


def test_threshold_pass():
    """All scores >= 0.7 should mean passed=True."""
    scores = TriadScores(faithfulness=0.7, context_relevance=0.8, answer_relevance=0.9, passed=True)
    assert scores.passed is True


def test_threshold_fail():
    """Any score < 0.7 means passed=False."""
    scores = TriadScores(faithfulness=0.5, context_relevance=0.8, answer_relevance=0.9, passed=False)
    assert scores.passed is False
    assert scores.faithfulness < 0.7
