from typing import Any
import streamlit as st

def render_graph(st: Any, dependency_graph: Any) -> None:
    """Render the dependency graph visualization."""
    st.markdown("### Architecture Summary")
    st.markdown(dependency_graph.get_architecture_summary())
    
    st.markdown("### Dependency Visualization")
    
    # Try to use pyvis for interactive graph
    try:
        from pyvis.network import Network
        import tempfile
        import streamlit.components.v1 as components
        
        # Filtering options
        st.markdown("**Filters**")
        col1, col2 = st.columns(2)
        show_files = col1.checkbox("Show Files", value=True)
        show_classes = col1.checkbox("Show Classes", value=True)
        show_functions = col2.checkbox("Show Functions", value=True)
        
        allowed_types = set()
        if show_files: allowed_types.add("file")
        if show_classes: allowed_types.add("class")
        if show_functions: allowed_types.add("function")
        
        # Create PyVis network
        net = Network(height='600px', width='100%', directed=True, 
                      bgcolor='#ffffff', font_color='black')
                      
        graph = dependency_graph.graph
        
        # Color mapping
        colors = {
            'file': 'lightblue',
            'class': 'lightgreen',
            'function': 'lightyellow'
        }
        
        # Add nodes
        for node_id, data in graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            if node_type in allowed_types:
                title = f"{node_type}: {node_id}"
                net.add_node(node_id, label=node_id, title=title, 
                             color=colors.get(node_type, 'gray'),
                             shape='dot' if node_type != 'file' else 'box')
                             
        # Edge colors based on relation
        edge_colors = {
            'imports': 'blue',
            'calls': 'orange',
            'inherits': 'red',
            'contains': 'gray'
        }
        
        # Add edges
        for u, v, data in graph.edges(data=True):
            if graph.nodes[u].get('node_type') in allowed_types and graph.nodes[v].get('node_type') in allowed_types:
                relation = data.get('relation', 'unknown')
                net.add_edge(u, v, title=relation, 
                             color=edge_colors.get(relation, 'black'))
                             
        # Generate and show HTML
        net.toggle_physics(True)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
            net.save_graph(tmp.name)
            with open(tmp.name, 'r', encoding='utf-8') as f:
                html = f.read()
            components.html(html, height=650)
            
    except ImportError:
        st.warning("PyVis is not installed. Showing raw JSON data instead.")
        st.json(dependency_graph.serialize())
