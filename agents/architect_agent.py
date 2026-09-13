"""Shared LLM generation for all specialist agents with retry and proper error reporting."""

from __future__ import annotations

from time import sleep, perf_counter
from typing import Any

from agents.state import AgentState, trace_event
from llm.provider import LLMProvider
from llm.utils import extract_text


ARCHITECT_SYSTEM_PROMPT = """You are a senior software architect. Analyze only the supplied repository evidence.
Identify architecture patterns, component structure, and dependency relationships.
Cite file paths and line numbers for every claim."""


class ArchitectAgent:
    """Repository architecture analysis backed by dependency graph context."""

    def __init__(self, provider: LLMProvider | None = None, dependency_graph: Any | None = None) -> None:
        self.llm_provider = provider or LLMProvider()
        self.dependency_graph = dependency_graph

    def node(self, state: AgentState) -> dict[str, Any]:
        started_at = perf_counter()
        graph_context = ""
        if self.dependency_graph:
            try:
                graph_context = self.dependency_graph.get_architecture_summary()
            except Exception:
                pass
        output = _generate(
            self.llm_provider,
            ARCHITECT_SYSTEM_PROMPT,
            state["query"],
            state.get("retrieved_chunks", []),
            graph_context,
        )
        return {
            "agent_outputs": {"architect": output},
            "agent_trace": [trace_event("architect", "analyzed architecture", started_at)],
        }


def _generate(
    provider: LLMProvider,
    system: str,
    query: str,
    chunks: list[dict[str, Any]],
    graph_context: str = "",
    max_retries: int = 2,
    agent_name: str = "agent",
) -> str:
    """Call the LLM with retry + backoff. Shows the real error on failure."""
    context = _format_context(chunks)
    prompt = [
        ("system", system),
        ("human", f"Question: {query}\n\nDependency graph:\n{graph_context}\n\nCode evidence:\n{context}"),
    ]

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            model = provider.get_chat_model("generation")
            response = model.invoke(prompt)
            return extract_text(response)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                wait = 2 ** attempt * 3  # 3s, 6s
                sleep(wait)

    # All retries exhausted — return useful error info, not just file list
    citations = _citations(chunks)
    error_type = type(last_error).__name__ if last_error else "Unknown"
    error_msg = str(last_error)[:200] if last_error else "No details"

    # Check for specific error types to give helpful messages
    if "429" in error_msg or "rate" in error_msg.lower() or "quota" in error_msg.lower() or "resource" in error_msg.lower():
        reason = (
            "**Your API rate limit has been exceeded.** "
            "The free tier of Gemini/Groq has limited requests per minute.\n\n"
            "**What you can do:**\n"
            "- ⏳ Wait 1-2 minutes and try again\n"
            "- 🔑 Add a Groq API key in the sidebar as a backup (free at https://console.groq.com)\n"
            "- 💳 Upgrade your Google API plan for higher rate limits"
        )
    elif "404" in error_msg or "NOT_FOUND" in error_msg:
        reason = f"Model not found. Check your model configuration. ({error_msg[:100]})"
    elif "401" in error_msg or "403" in error_msg or "PERMISSION" in error_msg:
        reason = "API key is invalid or expired. Update your API key in the sidebar."
    else:
        reason = f"{error_type}: {error_msg[:150]}"

    return (
        f"## ⚠️ LLM Generation Failed\n\n"
        f"All {max_retries + 1} attempts to generate a response failed.\n\n"
        f"{reason}\n\n"
        f"---\n\n"
        f"**Retrieved code evidence** ({len(chunks)} chunks): {', '.join(citations) or 'none'}"
    )


def _format_context(chunks: list[dict[str, Any]]) -> str:
    return "\n\n".join(f"[{chunk.get('metadata', {}).get('file_path', chunk.get('chunk_id', 'unknown'))}]\n{chunk.get('content', '')}" for chunk in chunks)


def _citations(chunks: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for chunk in chunks:
        path = chunk.get("metadata", {}).get("file_path", chunk.get("chunk_id", "?"))
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result
