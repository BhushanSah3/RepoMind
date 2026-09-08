"""RepoMind Streamlit application."""
import shutil
import time
from collections import Counter
from pathlib import Path

import streamlit as st

from config.settings import get_settings
from agents.code_retriever import CodeRetriever
from agents.graph import build_graph
from indexing.bm25_index import BM25Index
from indexing.embedder import LocalEmbedder
from indexing.vector_store import ChromaStore
from ingestion.ast_chunker import chunk_file
from ingestion.dependency_graph import DependencyGraph
from ingestion.file_walker import walk_repository
from ingestion.markdown_chunker import chunk_markdown
from ingestion.repo_cloner import clone_repo
from retrieval.contextual_retriever import ContextualRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.multi_query import MultiQueryExpander
from retrieval.reranker import Reranker
from ui.chat_interface import render_chat
from ui.graph_viewer import render_graph
from ui.repo_dashboard import render_dashboard

st.set_page_config(page_title='RepoMind', page_icon='🧠', layout='wide')
st.title('🧠 RepoMind')

for key, value in {'is_indexed': False, 'chat_history': []}.items(): 
    st.session_state.setdefault(key, value)

def ingest(url: str) -> None:
    settings = get_settings()
    
    parts = url.rstrip('/').split('/')
    slug = f"{parts[-2]}__{parts[-1]}" if len(parts) >= 2 else parts[-1]
    
    with st.status("Ingesting Repository", expanded=True) as status:
        st.write(f"**Cloning repository** `{url}`...")
        path = clone_repo(url)
        
        st.write("**Scanning files...**")
        files = walk_repository(path)
        language_counts = Counter(f.language for f in files)
        lang_str = ", ".join(f"{lang}: {count}" for lang, count in language_counts.items())
        st.write(f"Scanned {len(files)} files ({lang_str})")
        
        st.write("**AST parsing...**")
        chunks = []
        for file in files:
            if file.language == 'markdown':
                chunks.extend(chunk_markdown(file))
            else:
                chunks.extend(chunk_file(file))
        
        chunk_type_counts = Counter(getattr(c.chunk_type, 'value', c.chunk_type) for c in chunks)
        chunk_str = ", ".join(f"{ctype}: {count}" for ctype, count in chunk_type_counts.items())
        st.write(f"Created {len(chunks)} chunks ({chunk_str})")
        
        st.write("**Building dependency graph...**")
        graph = DependencyGraph()
        graph.build_from_chunks(chunks)
        st.write(f"Graph built with {graph.graph.number_of_nodes()} nodes and {graph.graph.number_of_edges()} edges")
        
        st.write("**Generating embeddings...**")
        start_time = time.time()
        embedder = LocalEmbedder()
        st.write(f"Embedding {len(chunks)} chunks × {settings.embedding_dimension} dimensions...")
        
        st.write(f"**Storing in ChromaDB...** (Collection: {slug})")
        chroma_dir = settings.chroma_dir
        store = ChromaStore(slug, chroma_dir, embedder)
        store.delete_collection()
        store.add_documents(chunks)
        end_time = time.time()
        st.write(f"Embeddings generated and stored in {end_time - start_time:.2f} seconds")
        
        st.write("**Building BM25 index...**")
        bm25 = BM25Index()
        bm25.build_index(chunks)
        token_count = sum(len(c.content.split()) for c in chunks)
        st.write(f"BM25 index built with ~{token_count} tokens")
        
        st.write("**Persisting BM25 and Dependency Graph to disk...**")
        bm25_path = settings.data_dir / 'bm25' / slug
        bm25_path.mkdir(parents=True, exist_ok=True)
        bm25.save(bm25_path / 'index.pkl')

        graph_path = settings.graph_dir / slug
        graph_path.mkdir(parents=True, exist_ok=True)
        graph.save(graph_path / 'graph.json')
        st.write(f"Saved artifacts to disk under slug `{slug}`")
        
        st.write("**Compiling LangGraph workflow...**")
        retriever = HybridRetriever(store, bm25, embedder)
        code_retriever = CodeRetriever(
            retriever, 
            query_expander=MultiQueryExpander(), 
            reranker=Reranker(), 
            contextual_retriever=ContextualRetriever(graph, chunks)
        )
        workflow = build_graph(code_retriever, dependency_graph=graph)
        st.write("Workflow compiled successfully")
        
        st.write("**Cleaning up cloned temp directory...**")
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass
            
        status.update(label="Ingestion Complete! ✅", state="complete", expanded=False)

    st.session_state.update(
        is_indexed=True, 
        files=files, 
        chunks=chunks, 
        dep_graph=graph, 
        stats=store.get_collection_stats(), 
        workflow=workflow,
        slug=slug
    )

with st.sidebar:
    st.subheader('Repository')
    url_input = st.text_input('GitHub URL', placeholder='https://github.com/owner/repo')
    
    if st.button('🔄 Ingest Repo', use_container_width=True):
        if url_input:
            try: 
                ingest(url_input)
            except Exception as error: 
                st.error(str(error))
        else:
            st.warning("Please enter a GitHub URL.")
            
    if st.session_state.is_indexed:
        st.divider()
        st.subheader("Index Statistics")
        st.metric("Files", len(st.session_state.files))
        st.metric("Chunks", len(st.session_state.chunks))
        
        lang_counts = len(set(f.language for f in st.session_state.files if hasattr(f, 'language') and f.language))
        st.metric("Languages", lang_counts)
        st.markdown(f"**Collection Slug:** `{st.session_state.slug}`")
        
    st.divider()
    with st.expander("⚙️ Settings"):
        try:
            settings = get_settings()
            if st.session_state.is_indexed:
                st.write(f"**Current LLM Provider:** {settings.default_llm_provider}")
            else:
                st.write(f"**LLM Provider:** {settings.default_llm_provider}")
        except Exception:
            st.write("**LLM Provider:** Unknown")

chat, dashboard, graph_tab = st.tabs(['Chat', 'Dashboard', 'Graph'])

with chat:
    if st.session_state.is_indexed: 
        render_chat(st, st.session_state.workflow)
    else: 
        st.info('Please index a repository first.')
        
with dashboard:
    if st.session_state.is_indexed: 
        render_dashboard(st, st.session_state.stats, st.session_state.files, st.session_state.get('chunks'))
        
with graph_tab:
    if st.session_state.is_indexed: 
        render_graph(st, st.session_state.dep_graph)
