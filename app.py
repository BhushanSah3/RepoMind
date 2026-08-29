"""RepoMind Streamlit application."""
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
for key, value in {'is_indexed':False, 'chat_history':[]}.items(): 
    st.session_state.setdefault(key, value)

def ingest(url: str) -> None:
    settings = get_settings()
    progress = st.progress(0, text='Cloning repository…')
    path = clone_repo(url)
    progress.progress(20, text='Scanning files…')
    
    files = walk_repository(path)
    chunks = []
    
    for number, file in enumerate(files, 1):
        chunks.extend(chunk_markdown(file) if file.language == 'markdown' else chunk_file(file))
        progress.progress(20 + int(35 * number / max(1,len(files))), text=f'Parsing files ({number}/{len(files)})…')
        
    progress.progress(60, text='Building dependency graph…')
    graph = DependencyGraph()
    graph.build_from_chunks(chunks)
    
    progress.progress(70, text='Generating local embeddings…')
    embedder = LocalEmbedder()
    chroma_dir = settings.chroma_dir
    
    store = ChromaStore(url.rsplit('/', 1)[-1], chroma_dir, embedder)
    store.delete_collection()
    store.add_documents(chunks)
    
    bm25 = BM25Index()
    bm25.build_index(chunks)
    retriever = HybridRetriever(store, bm25, embedder)
    
    code_retriever = CodeRetriever(
        retriever, 
        query_expander=MultiQueryExpander(), 
        reranker=Reranker(), 
        contextual_retriever=ContextualRetriever(graph, chunks)
    )
    
    st.session_state.update(
        is_indexed=True, 
        files=files, 
        chunks=chunks, 
        dep_graph=graph, 
        stats=store.get_collection_stats(), 
        workflow=build_graph(code_retriever)
    )
    progress.progress(100, text='Done ✅')

with st.sidebar:
    st.subheader('Repository')
    url = st.text_input('GitHub URL', placeholder='https://github.com/owner/repo')
    
    if st.button('🔄 Ingest Repo', use_container_width=True):
        try: 
            ingest(url)
        except Exception as error: 
            st.error(str(error))
            
    if st.session_state.is_indexed:
        st.divider()
        st.subheader("Index Statistics")
        st.metric("Files", len(st.session_state.files))
        st.metric("Chunks", len(st.session_state.chunks))
        
    st.divider()
    with st.expander("⚙️ Settings"):
        try:
            settings = get_settings()
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
