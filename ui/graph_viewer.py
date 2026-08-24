from typing import Any
def render_graph(st: Any, dependency_graph: Any) -> None:
    st.text(dependency_graph.get_architecture_summary())
    st.json(dependency_graph.serialize())
