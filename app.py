import streamlit as st
import os

from rag_logic.sidebar_handler import render_sidebar
from rag_logic.chat_handler import (
    setup_session_state, 
    render_chat_messages, 
    handle_user_input, 
    render_download_history,
    execute_ai_action
)
from rag_logic.llm_handler import get_llm_chain, get_summary_chain

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="RAG PDF Ultimate Portfolio", 
        page_icon="🤖", 
        layout="wide"
    )

    # Load custom CSS
    css_path = os.path.join("assets", "style.css")
    if os.path.exists(css_path):
        load_css(css_path)

    # Initialize session state
    setup_session_state()

    # Diagnostic Info (Hidden by default)
    if st.sidebar.checkbox("🔍 Debug Environment"):
        import sys
        import langchain
        st.sidebar.write(f"Python: {sys.version}")
        st.sidebar.write(f"LangChain: {langchain.__version__}")
        st.sidebar.write(f"Path: {sys.path[0]}")

    # Sidebar
    provider, model, api_key, backend = render_sidebar()

    # Main Content
    st.markdown('<h1 class="main-title">RAG ULTIMATE</h1>', unsafe_allow_html=True)
    st.markdown('<p class="dev-label">PROJECT BY TAHA ZAMAN</p>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

    # Tabbed Layout - ALWAYS VISIBLE
    tab1, tab2, tab3 = st.tabs(["💬 AI Conversation", "📂 Knowledge Workspace", "📊 Performance Analytics"])
    
    with tab1:
        if not st.session_state.processed:
            # Professional Introduction within the Chat Tab
            st.markdown("""
                <div class='glass-card' style='padding: 2rem; border-radius: 15px; background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px);'>
                    <h2 style='color: #00d4ff;'>👋 Welcome to RAG Ultimate</h2>
                    <p style='font-size: 1.1rem;'>Your intelligent document ecosystem is offline. <b>Upload your PDFs in the sidebar</b> to activate the Elite Engine.</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🔮 Intelligent Analysis")
                st.write("Using **LCEL** for lightning-fast document interrogation.")
                st.write("- **Dynamic Chunking**: Optimized character splitting.")
            with col2:
                st.subheader("🛠️ Technical Excellence")
                st.write("- **Hybrid Engine**: Toggle between memory and persistent storage.")
                st.write("- **Analytics Layer**: Real-time performance tracking.")
        else:
            # Active Chat Mode - Show a nice pulse to indicate "Ready"
            st.markdown(f"""
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; background: rgba(0, 212, 255, 0.05); padding: 10px; border-radius: 10px; border-left: 4px solid #00d4ff;'>
                    <div style='display: flex; align-items: center; gap: 10px;'>
                        <div style='width: 10px; height: 10px; background: #00ff88; border-radius: 50%; box-shadow: 0 0 8px #00ff88;'></div>
                        <span style='color: #00ff88; font-family: Courier; font-size: 0.9rem;'>ENGINE: ACTIVE</span>
                    </div>
                    <div style='color: #ccc; font-size: 0.8rem;'>
                        <b>CONTEXT:</b> {st.session_state.pdf_files[-1] if st.session_state.pdf_files else "Global"} ({st.session_state.workspace_mode})
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Quick Action Buttons below the pulse
            st.markdown("### ⚡ Elite Quick Actions")
            q1, q2, q3 = st.columns(3)
            with q1:
                if st.button("🔍 Summarize Risks", use_container_width=True):
                    st.session_state._quick_action = "Summarize the key risks or challenges mentioned in these documents."
            with q2:
                if st.button("📊 Extract Data", use_container_width=True):
                    st.session_state._quick_action = "Extract any statistical data or key metrics found in these files."
            with q3:
                if st.button("📝 Professional Brief", use_container_width=True):
                    st.session_state._quick_action = "Draft a professional one-paragraph brief based on the content."
            st.write("---")
            
            render_chat_messages()
            if st.session_state.vector_store:
                try:
                    chain = get_llm_chain(provider, model, st.session_state.vector_store, api_key)
                    # Handle Quick Action if triggered
                    if "_quick_action" in st.session_state:
                        prompt = st.session_state.pop("_quick_action")
                        execute_ai_action(chain, provider, model, prompt)

                    
                    handle_user_input(chain, provider, model)
                except Exception as e:
                    st.error(f"Chain error: {e}")


    with tab2:
        st.subheader("📚 Digital Knowledge Base & Index Stats")
        if st.session_state.get("pdf_files"):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.write("📂 **Indexed Documents**")
                for file in st.session_state.pdf_files:
                    st.success(f"✔️ {file}")
            
            with c2:
                st.write("📊 **Vector Statistics**")
                # Count total chunks if possible
                if st.session_state.vector_store:
                    # Generic way to get count for FAISS/Chroma
                    try:
                        if hasattr(st.session_state.vector_store, 'index'):
                            count = st.session_state.vector_store.index.ntotal
                        else:
                            count = st.session_state.vector_store._collection.count()
                        st.metric("Total Chunks", count)
                    except:
                        st.metric("Total Chunks", "N/A")
                
                st.caption(f"Backend: {backend}")
                st.caption(f"Embedding Provider: {provider}")
            
            st.divider()
            # Elite Executive Snapshot
            if st.session_state.get("doc_summary"):
                st.markdown("""
                    <div style='background: rgba(0, 212, 255, 0.05); padding: 20px; border-radius: 15px; border: 1px solid rgba(0, 212, 255, 0.2);'>
                        <h3 style='color: #00d4ff; margin-top: 0;'>📑 Executive Snapshot</h3>
                        <div style='font-size: 1.1rem; line-height: 1.6;'>
                """, unsafe_allow_html=True)
                st.markdown(st.session_state.doc_summary)
                st.markdown("</div></div><br>", unsafe_allow_html=True)

            st.info("Elite Engine is utilizing high-density vector transformations for context retrieval.")

        else:
            st.warning("No documents uploaded yet. Activation required.")

    with tab3:
        st.subheader("📊 Elite Performance Analytics")
        analytics = st.session_state.analytics
        
        # Calculate success rate based on successful queries vs total attempts
        success_rate = 100
        if analytics['total_queries'] > 0:
            success_rate = analytics.get("success_rate", 100)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Latency", f"{analytics['avg_latency']:.2f}s")
        m2.metric("Total Queries", analytics['total_queries'])
        m3.metric("Token Estimator", f"{analytics.get('total_tokens', 0):,}")
        m4.metric("Engine Status", "STABLE" if success_rate > 90 else "DEGRADED")

        
        if st.session_state.get("chat_history"):
            st.write("---")
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.write("📈 **Response Latency (Seconds)**")
                latencies = []
                for m in st.session_state.chat_history:
                    if "metadata" in m and "latency" in m["metadata"]:
                        try:
                            lat = float(m["metadata"]["latency"].replace('s',''))
                            latencies.append(lat)
                        except: pass
                if latencies:
                    st.area_chart(latencies, color="#00F2FE")

            with col_chart2:
                st.write("⏱️ **Session Activity**")
                # Placeholder for more complex activity stats
                if latencies:
                    st.line_chart(latencies, color="#92FE9D")

    # History Tools
    render_download_history()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p>© 2026 Developed by <b>Taha Zaman</b> | Built with Streamlit & LangChain</p>
            <p style='font-size: 0.8rem; letter-spacing: 1px;'>PREMIUM RAG PORTFOLIO EDITION</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
