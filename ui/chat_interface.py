from typing import Any
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from ui.agent_trace import render_agent_trace
from ui.components import score_badge

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
            with st.spinner('Analyzing repository...'):
                result = workflow.invoke({
                    'query': query,
                    'chat_history': lc_history,
                    'retry_count': 0,
                    'agent_trace': []
                })
                
            st.markdown(result['synthesized_answer'])
            
            # Show agent trace
            render_agent_trace(st, result.get('agent_trace', []), result.get('scores', None))
            
            # Show RAG Triad scores if present directly in the chat message
            if 'scores' in result and result['scores']:
                st.markdown("---")
                scores = result['scores']
                cols = st.columns(3)
                if 'faithfulness' in scores:
                    cols[0].markdown(score_badge(scores['faithfulness'], "Faithfulness"))
                if 'context_relevance' in scores:
                    cols[1].markdown(score_badge(scores['context_relevance'], "Context Relevance"))
                if 'answer_relevance' in scores:
                    cols[2].markdown(score_badge(scores['answer_relevance'], "Answer Relevance"))
                    
        # Update chat history
        st.session_state.chat_history.extend([
            {'role': 'user', 'content': query},
            {'role': 'assistant', 'content': result['synthesized_answer']}
        ])
