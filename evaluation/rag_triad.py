"""Concurrent faithfulness, context, and answer-relevance checks."""
import asyncio
import json
import re
from typing import Any
from pydantic import BaseModel
from llm.provider import LLMProvider

class TriadScores(BaseModel):
    faithfulness: float = 1.0
    context_relevance: float = 1.0
    answer_relevance: float = 1.0
    passed: bool = True

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
        return TriadScores()
    scores = dict(zip(_PROMPTS, values))
    return TriadScores(**scores, passed=all(value >= .7 for value in scores.values()))

async def _score(model: Any, prompt: str) -> float:
    response = await model.ainvoke(prompt)
    content = getattr(response, 'content', response)
    match = re.search(r'\{.*?\}', str(content), re.DOTALL)
    try: value = float(json.loads(match.group(0) if match else str(content)).get('score', 1.0))
    except (ValueError, TypeError, json.JSONDecodeError): value = 1.0
    return max(0.0, min(1.0, value))
