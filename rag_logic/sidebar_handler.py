import streamlit as st
from rag_logic.config import MODEL_OPTIONS, VECTOR_BACKENDS
from rag_logic.pdf_handler import get_pdf_documents, get_text_chunks
from rag_logic.vector_handler import create_vectorstore, load_local_vectorstore

def render_sidebar():
    """
    Renders the sidebar with professional branding and configuration.
    """
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align: center; padding: 1rem;'>
                <h2 style='margin:0; font-family: Outfit;'>TZ</h2>
                <small style='letter-spacing: 2px;'>TAHA ZAMAN</small>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.divider()
        
        with st.expander("⚙️ AI CONFIGURATION", expanded=True):
            provider = st.selectbox("Provider", list(MODEL_OPTIONS.keys()), key="provider")
            api_key = st.text_input("API Key", type="password", value=st.session_state.get("api_key", ""))
            
            model = st.selectbox("Model", MODEL_OPTIONS[provider]["models"], key="model")
            backend = st.selectbox("Vector Backend", VECTOR_BACKENDS, key="backend")
            
        with st.expander("📚 Knowledge Base", expanded=True):
            st.session_state.workspace_mode = st.radio(
                "Workspace Mode",
                ["Active Document", "Full Corpus"],
                help="Active Document: Only searches the latest upload. Full Corpus: Searches all indexed files.",
                horizontal=True
            )
            
            uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
            
            if st.button("✨ Process & Vectorize", use_container_width=True):
                if uploaded_files and api_key:
                    with st.spinner("Analyzing documents..."):
                        # Get Metadata-Aware Documents
                        documents = get_pdf_documents(uploaded_files)
                        chunks = get_text_chunks(documents)
                        
                        if chunks:
                            # Handle Workspace Mode logic
                            if st.session_state.workspace_mode == "Active Document":
                                st.session_state.pdf_files = [f.name for f in uploaded_files]
                                vectorstore = create_vectorstore(chunks, provider, api_key, backend)
                            
                            st.session_state.vector_store = vectorstore
                            st.session_state.processed = True
                            
                            # Generate Executive Snapshot
                            from rag_logic.llm_handler import get_summary_chain
                            with st.spinner("Generating Executive Snapshot..."):
                                try:
                                    summary_chain = get_summary_chain(provider, model, vectorstore, api_key)
                                    st.session_state.doc_summary = summary_chain.invoke("Generate summary")
                                except Exception as e:
                                    st.session_state.doc_summary = f"Summary unavailable: {e}"
                                    
                            st.success("Documents vectorized & Snapshot Generated!")
                        else:
                            st.error("❌ No readable text found in the uploaded PDFs. Please ensure they are not scanned images or empty.")
                elif not api_key:
                    st.error("Please enter an API Key.")
                else:
                    st.warning("No files uploaded.")

        with st.expander("📊 WORKSPACE INSIGHTS"):
            if st.session_state.get("pdf_files"):
                st.write(f"📂 **Active Corpus**: {len(st.session_state.pdf_files)} Files")
                st.caption(f"Mode: {st.session_state.workspace_mode}")
                st.progress(100, text="Knowledge Synchronized")
            else:
                st.caption("Knowledge base empty.")

        with st.expander("🛠️ ELITE TOOLS"):
            if st.button("🆕 New Chat Session", use_container_width=True, type="primary"):
                st.session_state.chat_history = []
                st.rerun()

            st.session_state.dev_mode = st.toggle("🚀 Developer Insights", help="Show raw retrieval data and chain logic")
            
            if st.button("🔄 Full System Reset", use_container_width=True):
                st.session_state.clear()
                st.rerun()


    return provider, model, api_key, backend
