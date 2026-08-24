from typing import Any
from ui.agent_trace import render_agent_trace
def render_chat(st: Any, workflow: Any) -> None:
    for message in st.session_state.get('chat_history', []):
        with st.chat_message(message['role']): st.markdown(message['content'])
    if query := st.chat_input('Ask about the indexed repository'):
        with st.chat_message('user'): st.markdown(query)
        with st.chat_message('assistant'):
            result = workflow.invoke({'query': query, 'chat_history': [], 'retry_count': 0, 'agent_trace': []})
            st.markdown(result['synthesized_answer']); render_agent_trace(st, result.get('agent_trace', []))
        st.session_state.chat_history.extend([{'role':'user','content':query},{'role':'assistant','content':result['synthesized_answer']}])
