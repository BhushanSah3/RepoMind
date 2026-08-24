from typing import Any
def render_agent_trace(st: Any, trace: list[dict]) -> None:
    with st.expander('🔍 Agent Trace'):
        for event in trace: st.write(f"**{event.get('agent', 'system')}** — {event.get('action', '')}")
