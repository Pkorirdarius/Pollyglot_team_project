"""
ui/streamlit_app.py
Updated for 2026 standards: Enhanced source tracking and robust state handling.
"""
from __future__ import annotations
import sys
import os
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from config.settings import settings
from data_wrangling.loader import load_and_split
from data_wrangling.vectorstore import ingest_documents
from search.models.rag_pipeline import run_rag_query
from search.models.schemas import QueryRequest

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Polyglot RAG", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner chat interface
st.markdown("""
    <style>
        .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
        .source-tag { font-size: 0.8rem; color: #888; }
    </style>
""", unsafe_content_usage=True)

st.title("🤖 Polyglot RAG Assistant")
st.caption(f"🚀 Powered by **{settings.llm_provider.upper()}** ({settings.llm_model})")

# ── Sidebar: Data Management ──────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Knowledge Base")
    uploaded_file = st.file_uploader(
        "Upload reference documents", 
        type=["pdf", "txt", "docx", "csv"],
        help="Files will be chunked and stored in your local vector store."
    )
    
    if uploaded_file and st.button("✨ Ingest Data", use_container_width=True):
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        
        source_type = suffix.lstrip(".")
        with st.status("Processing document...", expanded=True) as status:
            try:
                st.write("Reading file...")
                chunks = load_and_split(tmp_path, source_type=source_type)
                st.write(f"Generating embeddings for {len(chunks)} chunks...")
                n = ingest_documents(chunks)
                status.update(label="Ingestion Complete!", state="complete", expanded=False)
                st.toast(f"Successfully added {n} chunks!", icon="✅")
            except Exception as exc:
                status.update(label="Ingestion Failed", state="error")
                st.error(f"Details: {exc}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

    st.divider()
    st.header("⚙️ Retrieval Settings")
    top_k = st.slider("Context chunks (Top-K)", 1, 15, settings.top_k_retrieval)
    st.info("Higher Top-K provides more context but increases latency.")

# ── Chat Interface ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# New Input
if prompt := st.chat_input("Ask a question about your data..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant message
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner(f"{settings.llm_model} is thinking..."):
            try:
                req = QueryRequest(query=prompt, top_k=top_k)
                resp = run_rag_query(req)
                
                # Render Answer
                response_placeholder.markdown(resp.answer)
                
                # Render Sources
                if resp.sources:
                    with st.expander(f"📚 View Citations ({len(resp.sources)})"):
                        for i, chunk in enumerate(resp.sources, 1):
                            page_info = f", Page {chunk.page}" if chunk.page else ""
                            st.markdown(f"**{i}. {chunk.source}**{page_info} `score: {chunk.score:.3f}`")
                            st.caption(f"\"{chunk.text[:300]}...\"")
                            st.divider()
                
                st.session_state.messages.append({"role": "assistant", "content": resp.answer})
                
            except Exception as exc:
                error_msg = f"⚠️ **Error during generation:** {str(exc)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})