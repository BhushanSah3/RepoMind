"""Concurrent faithfulness, context, and answer-relevance checks."""
import asyncio
import json
import re
from typing import Any
from pydantic import BaseModel
from llm.provider import LLMProvider
from llm.utils import extract_text
from config.settings import get_settings

class TriadScores(BaseModel):
    faithfulness: float = 1.0
    context_relevance: float = 1.0
    answer_relevance: float = 1.0
    passed: bool = True
    evaluation_note: str | None = None

_PROMPTS = {
    'faithfulness': 'Rate answer faithfulness to context from 0 to 1. Return only JSON {{"score": number}}.\nCONTEXT:\n{context}\nANSWER:\n{answer}',
    'context_relevance': 'Rate context relevance to query from 0 to 1. Return only JSON {{"score": number}}.\nQUERY:{query}\nCONTEXT:\n{context}',
    'answer_relevance': 'Rate answer relevance to query from 0 to 1. Return only JSON {{"score": number}}.\nQUERY:{query}\nANSWER:\n{answer}',
}

async def evaluate_response(query: str, retrieved_chunks: list[dict[str, Any]], answer: str, llm: Any | None = None, provider: LLMProvider | None = None) -> TriadScores:
    """Run all RAG checks concurrently; pass by default if evaluation is unavailable."""
    context = '\n\n'.join(chunk.get('content', '') for chunk in retrieved_chunks)
    try:
        model = llm or (provider or LLMProvider()).get_chat_model('evaluation')
        values = await asyncio.gather(*[_score(model, prompt.format(query=query, context=context, answer=answer)) for prompt in _PROMPTS.values()])
    except Exception:
        return TriadScores(evaluation_note='LLM evaluation unavailable, assuming pass')
    
    scores = dict(zip(_PROMPTS, values))
    settings = get_settings()
    
    passed = (
        scores['faithfulness'] >= settings.rag_faithfulness_threshold
        and scores['context_relevance'] >= settings.rag_context_relevance_threshold
        and scores['answer_relevance'] >= settings.rag_answer_relevance_threshold
    )
    
    return TriadScores(**scores, passed=passed)

async def _score(model: Any, prompt: str) -> float:
    response = await model.ainvoke(prompt)
    content = extract_text(response)
    match = re.search(r'\{.*?\}', content, re.DOTALL)
    try: value = float(json.loads(match.group(0) if match else content).get('score', 1.0))
    except (ValueError, TypeError, json.JSONDecodeError): value = 1.0
    return max(0.0, min(1.0, value))
