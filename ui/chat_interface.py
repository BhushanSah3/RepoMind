from typing import Any
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from ui.agent_trace import render_agent_trace
from ui.components import score_badge, format_duration

def render_chat(st: Any, workflow: Any) -> None:
    """Render the chat interface."""
    chat_history = st.session_state.get('chat_history', [])
    
    for message in chat_history:
        role = message['role']
        avatar = '👤' if role == 'user' else '🧠'
        with st.chat_message(role, avatar=avatar):
            st.markdown(message['content'])
            
    if query := st.chat_input('Ask about the indexed repository'):
        # Add user message to history
        with st.chat_message('user', avatar='👤'):
            st.markdown(query)
            
        # Convert chat history to LangChain messages format, limited to last 10
        lc_history = []
        for msg in chat_history[-10:]:
            if msg['role'] == 'user':
                lc_history.append(HumanMessage(content=msg['content']))
            else:
                lc_history.append(AIMessage(content=msg['content']))
                
        with st.chat_message('assistant', avatar='🧠'):
            with st.status('🧠 Processing query...', expanded=True) as status:
                st.write("Initializing workflow...")
                result = workflow.invoke({
                    'query': query,
                    'chat_history': lc_history,
                    'retry_count': 0,
                    'agent_trace': []
                })
                
                # Analyze trace for timings
                trace = result.get('agent_trace', [])
                total_duration = sum(t.get('duration_ms', 0) for t in trace)
                
                st.markdown(f"**Query Analyzer**: Intent = `{result.get('intent', 'unknown')}`, Scope = `{result.get('scope', 'unknown')}`")
                
                chunks = result.get('retrieved_chunks', [])
                st.markdown(f"**Hybrid Search**: {len(chunks)} chunks, Strategy = `{result.get('retrieval_strategy', 'default')}`")
                st.markdown(f"**Reranking**: Top-{len(chunks)} selected")
                
                agents = result.get('target_agents', [])
                st.markdown(f"**Agent(s)**: {', '.join(agents) if agents else 'None'}")
                
                st.markdown("**RAG Triad**:")
                cols_status = st.columns(3)
                cols_status[0].markdown(score_badge(result.get('faithfulness_score', 1.0), "Faith"), unsafe_allow_html=True)
                cols_status[1].markdown(score_badge(result.get('context_relevance_score', 1.0), "Context"), unsafe_allow_html=True)
                cols_status[2].markdown(score_badge(result.get('answer_relevance_score', 1.0), "Answer"), unsafe_allow_html=True)
                
                st.markdown(f"**Total time**: {format_duration(total_duration)}")
                
                status.update(label=f"Query processed in {format_duration(total_duration)}", state="complete", expanded=False)
                
            # Display answer
            st.markdown(result.get('synthesized_answer', ''))
            
            # Show agent trace
            render_agent_trace(st, result.get('agent_trace', []), result)
            
            # Show RAG Triad scores
            st.markdown("---")
            cols = st.columns(3)
            cols[0].markdown(score_badge(result.get('faithfulness_score', 1.0), "Faithfulness"), unsafe_allow_html=True)
            cols[1].markdown(score_badge(result.get('context_relevance_score', 1.0), "Context Relevance"), unsafe_allow_html=True)
            cols[2].markdown(score_badge(result.get('answer_relevance_score', 1.0), "Answer Relevance"), unsafe_allow_html=True)
            
            if result.get('retry_count', 0) >= 2 and not result.get('evaluation_passed', True):
                st.warning("⚠️ Confidence Warning: The evaluation failed after maximum retries. The answer might not be fully supported by the context.")
                    
        # Update chat history
        st.session_state.chat_history.extend([
            {'role': 'user', 'content': query},
            {'role': 'assistant', 'content': result.get('synthesized_answer', '')}
        ])
