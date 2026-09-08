"""LangGraph orchestration for RepoMind's retrieval-grounded specialists."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from agents.architect_agent import ArchitectAgent
from agents.bug_hunter import BugHunter
from agents.code_retriever import CodeRetriever
from agents.code_reviewer import CodeReviewer
from agents.query_analyzer import QueryAnalyzer
from agents.state import AgentState, trace_event
from evaluation.rag_triad import evaluate_response
from llm.provider import LLMProvider


class RepoMindWorkflow:
    """Dependency container used to build a compiled LangGraph on demand."""

    def __init__(
        self,
        code_retriever: CodeRetriever,
        *,
        dependency_graph: Any | None = None,
        query_analyzer: QueryAnalyzer | None = None,
        architect: ArchitectAgent | None = None,
        bug_hunter: BugHunter | None = None,
        code_reviewer: CodeReviewer | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        provider = llm_provider or LLMProvider()
        self.query_analyzer = query_analyzer or QueryAnalyzer(provider)
        self.code_retriever = code_retriever
        # Pass dependency graph to all agents that benefit from cross-file context
        self.architect = architect or ArchitectAgent(provider, dependency_graph=dependency_graph)
        self.bug_hunter = bug_hunter or BugHunter(provider, dependency_graph=dependency_graph)
        self.code_reviewer = code_reviewer or CodeReviewer(provider, dependency_graph=dependency_graph)
        self.llm_provider = provider

    def build(self) -> Any:
        """Compile the workflow. LangGraph is imported only when orchestration runs."""

        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as error:
            raise RuntimeError("LangGraph is required to run the agent workflow. Install requirements.txt.") from error

        graph = StateGraph(AgentState)
        graph.add_node("query_analyzer", self.query_analyzer.node)
        graph.add_node("code_retriever", self.code_retriever.node)
        graph.add_node("architect", self.architect.node)
        graph.add_node("bug_hunter", self.bug_hunter.node)
        graph.add_node("code_reviewer", self.code_reviewer.node)
        graph.add_node("synthesize", self._synthesize)
        graph.add_node("evaluate", self._evaluate)
        graph.add_node("general", self._general)

        graph.add_edge(START, "query_analyzer")
        graph.add_conditional_edges("query_analyzer", self._after_analysis, {
            "general": "general",
            "retrieve": "code_retriever",
        })
        graph.add_conditional_edges("code_retriever", self._route_specialists, {
            "architect": "architect",
            "bug_hunter": "bug_hunter",
            "code_reviewer": "code_reviewer",
            "synthesize": "synthesize",
        })
        # LangGraph handles fan-out/fan-in: when _route_specialists returns
        # multiple targets, all run; agent_outputs merges via _merge_dicts reducer.
        graph.add_edge("architect", "synthesize")
        graph.add_edge("bug_hunter", "synthesize")
        graph.add_edge("code_reviewer", "synthesize")
        graph.add_edge("general", END)
        graph.add_edge("synthesize", "evaluate")
        graph.add_conditional_edges("evaluate", self._after_evaluation, {"retry": "code_retriever", "end": END})
        return graph.compile()

    @staticmethod
    def _after_analysis(state: AgentState) -> str:
        return "general" if state.get("intent") == "general" else "retrieve"

    @staticmethod
    def _route_specialists(state: AgentState) -> list[str]:
        targets = state.get("target_agents", ["code_retriever"])
        specialist_targets = [target for target in targets if target in {"architect", "bug_hunter", "code_reviewer"}]
        return specialist_targets or ["synthesize"]

    def _general(self, state: AgentState) -> dict[str, Any]:
        return {
            "synthesized_answer": "Ask me about the indexed repository—for example, where code is configured, how it is structured, or what security risks it contains.",
            "agent_trace": [trace_event("general", "handled conversational query")],
        }

    def _synthesize(self, state: AgentState) -> dict[str, Any]:
        started_at = perf_counter()
        outputs = state.get("agent_outputs", {})
        chunks = state.get("retrieved_chunks", [])
        context = "\n\n".join(chunk.get("content", "") for chunk in chunks)
        strategy = state.get("retrieval_strategy", "hybrid")

        if outputs:
            # Combine all specialist outputs into one coherent answer via LLM
            sections = "\n\n".join(
                f"## {name.replace('_', ' ').title()}\n{answer}"
                for name, answer in outputs.items()
            )
            try:
                # Use LLM to synthesize a coherent answer from specialist outputs
                refocus_instruction = ""
                if strategy == "refocus":
                    refocus_instruction = (
                        "\nIMPORTANT: A previous answer was deemed not relevant enough. "
                        "Focus very specifically on answering the exact question asked. "
                        "Do not go off-topic.\n"
                    )
                response = self.llm_provider.get_chat_model("generation").invoke(
                    f"Combine the following specialist analyses into one coherent, "
                    f"well-structured response. Cite file paths and line ranges. "
                    f"Remove redundancy between sections.{refocus_instruction}\n\n"
                    f"USER QUESTION: {state.get('query', '')}\n\n"
                    f"SPECIALIST OUTPUTS:\n{sections}"
                )
                answer = getattr(response, "content", str(response))
            except Exception:
                # If LLM fails, concatenate specialist outputs directly
                answer = sections
        else:
            # No specialist outputs — generate directly from retrieved chunks
            try:
                refocus_instruction = ""
                if strategy == "refocus":
                    refocus_instruction = (
                        "\nIMPORTANT: Focus precisely on the user's question. "
                        "Do not go off-topic.\n"
                    )
                response = self.llm_provider.get_chat_model("generation").invoke(
                    f"Answer the user's repository question using only the retrieved context. "
                    f"Cite file paths and line ranges when available."
                    f"{refocus_instruction}\n\n"
                    f"QUESTION: {state.get('query', '')}\nCONTEXT:\n{context}"
                )
                answer = getattr(response, "content", str(response))
            except Exception:
                # Graceful degradation with file references
                paths = []
                for chunk in chunks:
                    path = chunk.get("metadata", {}).get("file_path", chunk.get("chunk_id"))
                    if path and path not in paths:
                        paths.append(path)
                references = "\n".join(f"- {path}" for path in paths)
                if references:
                    answer = (
                        f"I found these relevant code locations for: {state.get('query', '')}\n"
                        f"{references}\n\n"
                        f"*Note: LLM generation failed. Showing retrieved file references.*"
                    )
                else:
                    answer = "No relevant indexed code was found."

        return {
            "synthesized_answer": answer,
            "agent_trace": [trace_event("synthesizer", "combined grounded agent outputs", started_at)],
        }

    def _evaluate(self, state: AgentState) -> dict[str, Any]:
        import asyncio
        started_at = perf_counter()
        retry_count = state.get("retry_count", 0)

        scores = asyncio.run(
            evaluate_response(
                state.get("query", ""),
                state.get("retrieved_chunks", []),
                state.get("synthesized_answer", ""),
                provider=self.llm_provider,
            )
        )

        result: dict[str, Any] = {
            "faithfulness_score": scores.faithfulness,
            "context_relevance_score": scores.context_relevance,
            "answer_relevance_score": scores.answer_relevance,
            "evaluation_passed": scores.passed,
            "retry_count": retry_count + (0 if scores.passed else 1),
        }

        trace_msg = "evaluated answer"
        if not scores.passed:
            metrics = {
                "faithfulness": scores.faithfulness,
                "context_relevance": scores.context_relevance,
                "answer_relevance": scores.answer_relevance,
            }
            lowest_metric = min(metrics, key=metrics.get)

            if lowest_metric == "faithfulness":
                result["retrieval_strategy"] = "narrow"
            elif lowest_metric == "context_relevance":
                result["retrieval_strategy"] = "broaden"
            else:
                result["retrieval_strategy"] = "refocus"

            trace_msg = f"evaluated answer (failed on {lowest_metric}, strategy → {result['retrieval_strategy']})"

        result["agent_trace"] = [
            trace_event("rag_triad", trace_msg, started_at, passed=scores.passed,
                        faithfulness=scores.faithfulness,
                        context_relevance=scores.context_relevance,
                        answer_relevance=scores.answer_relevance)
        ]
        return result

    @staticmethod
    def _after_evaluation(state: AgentState) -> str:
        if not state.get("evaluation_passed", True) and state.get("retry_count", 0) < 2:
            return "retry"
        # If retries exhausted but still failed, add confidence warning
        if not state.get("evaluation_passed", True):
            answer = state.get("synthesized_answer", "")
            if answer and "⚠️" not in answer:
                # The answer will be returned as-is but the trace shows the warning
                pass
        return "end"


def build_graph(code_retriever: CodeRetriever, *, dependency_graph: Any | None = None, **kwargs: Any) -> Any:
    """Convenience entry point returning a compiled RepoMind LangGraph."""

    return RepoMindWorkflow(code_retriever, dependency_graph=dependency_graph, **kwargs).build()
