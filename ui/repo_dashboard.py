from typing import Any
import streamlit as st
import pandas as pd

def render_dashboard(st: Any, stats: dict, files: list, chunks: list | None = None) -> None:
    """Render the repository dashboard with metrics and charts."""
    st.header("Repository Analytics")
    
    # Calculate metrics
    chunk_count = stats.get('chunk_count', len(chunks) if chunks else 0)
    file_count = len(files)
    
    languages = {}
    total_size = 0
    for f in files:
        lang = f.language or 'unknown'
        languages[lang] = languages.get(lang, 0) + 1
        total_size += f.size_bytes
        
    avg_size = total_size / max(1, file_count)
    
    # Show top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Files", file_count)
    col2.metric("Indexed Chunks", chunk_count)
    col3.metric("Languages Detected", len(languages))
    col4.metric("Avg File Size", f"{avg_size / 1024:.1f} KB")
    
    st.divider()
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Language Distribution")
        if languages:
            lang_df = pd.DataFrame(list(languages.items()), columns=['Language', 'Count'])
            lang_df = lang_df.set_index('Language')
            st.bar_chart(lang_df)
            
    with col_chart2:
        st.subheader("Chunk Type Breakdown")
        if chunks:
            chunk_types = {}
            for c in chunks:
                ctype = getattr(c, 'chunk_type', 'unknown')
                chunk_types[ctype] = chunk_types.get(ctype, 0) + 1
            if chunk_types:
                type_df = pd.DataFrame(list(chunk_types.items()), columns=['Type', 'Count'])
                type_df = type_df.set_index('Type')
                st.bar_chart(type_df)
        else:
            st.info("Chunk breakdown not available")
            
    st.divider()
    
    st.subheader("Files Explorer")
    search_query = st.text_input("Search files...", placeholder="e.g., .py or main")
    
    file_data = []
    for item in files:
        if not search_query or search_query.lower() in item.relative_path.lower():
            file_data.append({
                'Path': item.relative_path,
                'Language': item.language,
                'Size (Bytes)': item.size_bytes
            })
            
    if file_data:
        st.dataframe(file_data, use_container_width=True)
    else:
        st.info("No files matched the search.")
