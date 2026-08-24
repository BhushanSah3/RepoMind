"""LLM-backed, structured query expansion for code retrieval."""

import logging
from typing import Any

from pydantic import BaseModel, Field

from llm.provider import LLMProvider


logger = logging.getLogger(__name__)


MULTI_QUERY_PROMPT = """You are a code search query expander. Given a user's question
about a codebase, generate 3 alternative search queries that would help find relevant code.

Rules:
- Each query should approach the question from a different angle.
- Include both semantic (natural language) and keyword (code terms) variants.
- Keep queries concise (under 20 words each).

User question: {query}
"""


class QueryExpansion(BaseModel):
    """Structured alternatives returned by the query-expansion model."""

    queries: list[str] = Field(min_length=3, max_length=3)


class MultiQueryExpander:
    """Generate concise, distinct query variants without blocking retrieval on LLM failure."""

    def __init__(self, llm: Any | None = None, provider: LLMProvider | None = None) -> None:
        self._llm = llm
        self._provider = provider

    def expand(self, original_query: str) -> list[str]:
        """Return up to three distinct alternatives, falling back to the original query."""

        query = original_query.strip()
        if not query:
            return []

        try:
            structured_llm = self._get_llm().with_structured_output(QueryExpansion)
            response = structured_llm.invoke(MULTI_QUERY_PROMPT.format(query=query))
            expansion = response if isinstance(response, QueryExpansion) else QueryExpansion.model_validate(response)
            queries = _unique_queries(expansion.queries, original=query)
            if len(queries) == 3:
                return queries
            logger.warning("Query expansion returned fewer than three distinct variants.")
        except Exception as error:  # LLM failures should not prevent a local search.
            logger.warning("Query expansion failed; using the original query: %s", error)
        return [query]

    def _get_llm(self) -> Any:
        if self._llm is None:
            self._llm = (self._provider or LLMProvider()).get_chat_model("expansion")
        return self._llm


def expand_query(original_query: str, llm: Any | None = None) -> list[str]:
    """Convenience API for one-off expansion calls."""

    return MultiQueryExpander(llm=llm).expand(original_query)


def _unique_queries(queries: list[str], original: str) -> list[str]:
    seen = {original.casefold()}
    cleaned: list[str] = []
    for query in queries:
        normalized = " ".join(query.split())
        if normalized and normalized.casefold() not in seen:
            seen.add(normalized.casefold())
            cleaned.append(normalized)
    return cleaned
