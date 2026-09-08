from typing import Any
import streamlit as st
from ui.components import agent_icon, format_duration, score_badge

def render_agent_trace(st: Any, trace: list[dict], result: dict | None = None) -> None:
    """Render the agent execution trace timeline."""
    if not trace and not result:
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
                st.markdown(header)
                st.json(details, expanded=False)
            else:
                st.markdown(header)
                
        if result:
            st.markdown("### Execution Summary")
            
            chunks = result.get('retrieved_chunks', [])
            agents = result.get('target_agents', [])
            retry_count = result.get('retry_count', 0)
            
            st.markdown(f"**Retrieved Chunks:** {len(chunks)}")
            st.markdown(f"**Agents Activated:** {', '.join(agents) if agents else 'None'}")
            
            if retry_count > 0:
                st.warning(f"Response generated after {retry_count} retries based on evaluation.")
                
            st.markdown("### RAG Triad Scores")
            cols = st.columns(3)
            
            faith = result.get('faithfulness_score', 1.0)
            context = result.get('context_relevance_score', 1.0)
            answer = result.get('answer_relevance_score', 1.0)
            
            cols[0].markdown(score_badge(faith, "Faithfulness"), unsafe_allow_html=True)
            cols[1].markdown(score_badge(context, "Context Relevance"), unsafe_allow_html=True)
            cols[2].markdown(score_badge(answer, "Answer Relevance"), unsafe_allow_html=True)
                
        st.markdown(f"**Total Execution Time:** {format_duration(total_time)}")
