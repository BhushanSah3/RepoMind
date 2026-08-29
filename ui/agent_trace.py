from typing import Any
import streamlit as st
from ui.components import agent_icon, format_duration, score_badge

def render_agent_trace(st: Any, trace: list[dict], scores: dict | None = None) -> None:
    """Render the agent execution trace timeline."""
    if not trace and not scores:
        return
        
    with st.expander('🔍 Agent Trace'):
        total_time = 0
        
        for event in trace:
            agent = event.get('agent', 'system')
            action = event.get('action', '')
            duration = event.get('duration_ms')
            details = event.get('details')
            
            icon = agent_icon(agent)
            
            header = f"{icon} **{agent}** — {action}"
            if duration is not None:
                total_time += duration
                header += f" ({format_duration(duration)})"
                
            if details:
                # If there are details, show them in an expander inside
                # But since we are already in an expander, we can use a container or raw text
                st.markdown(header)
                st.json(details, expanded=False)
            else:
                st.markdown(header)
                
        if scores:
            st.markdown("### RAG Triad Scores")
            cols = st.columns(3)
            
            if 'faithfulness' in scores:
                cols[0].markdown(score_badge(scores['faithfulness'], "Faithfulness"))
            if 'context_relevance' in scores:
                cols[1].markdown(score_badge(scores['context_relevance'], "Context"))
            if 'answer_relevance' in scores:
                cols[2].markdown(score_badge(scores['answer_relevance'], "Answer"))
                
            if scores.get('retry_count', 0) > 0:
                st.warning(f"Response generated after {scores['retry_count']} retries based on evaluation.")
                
        st.markdown(f"**Total Execution Time:** {format_duration(total_time)}")
