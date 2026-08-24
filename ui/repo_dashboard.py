from typing import Any
def render_dashboard(st: Any, stats: dict, files: list) -> None:
    st.metric('Indexed chunks', stats.get('chunk_count', 0)); st.metric('Files', len(files))
    st.dataframe([{'path': item.relative_path, 'language': item.language, 'bytes': item.size_bytes} for item in files], use_container_width=True)
