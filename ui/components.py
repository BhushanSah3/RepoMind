"""Reusable Streamlit components and widgets."""

def score_badge(score: float, label: str) -> str:
    """Returns emoji+color indicator for a score."""
    if score >= 0.8:
        emoji = "🟢"
    elif score >= 0.7:
        emoji = "🟡"
    else:
        emoji = "🔴"
    return f"{emoji} **{label}:** {score:.2f}"

def agent_icon(agent_name: str) -> str:
    """Returns icon mapping for agent types."""
    icons = {
        "query_analyzer": "🧠",
        "code_retriever": "🔍",
        "architect": "🏗️",
        "bug_hunter": "🐛",
        "code_reviewer": "📋",
        "synthesizer": "📝",
        "rag_triad": "📊",
        "general": "💬"
    }
    return icons.get(agent_name.lower(), "💬")

def format_duration(ms: float) -> str:
    """Format duration in milliseconds to human-readable string."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes}m {seconds:.1f}s"

def code_block(code: str, language: str = 'python') -> str:
    """Format code block as markdown."""
    return f"```{language}\n{code}\n```"
