import streamlit as st
import pandas as pd
from datetime import datetime

def setup_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "analytics" not in st.session_state:
        st.session_state.analytics = {
            "total_queries": 0,
            "avg_latency": 0.0,
            "total_tokens": 0,
            "success_rate": 100
        }
    if "dev_mode" not in st.session_state:
        st.session_state.dev_mode = False
    if "workspace_mode" not in st.session_state:
        st.session_state.workspace_mode = "Active Document"
    if "pdf_files" not in st.session_state:
        st.session_state.pdf_files = []
    if "auto_summary_requested" not in st.session_state:
        st.session_state.auto_summary_requested = False
    if "doc_summary" not in st.session_state:
        st.session_state.doc_summary = None

def estimate_tokens(text):
    """Simple heuristic for token estimation (4 chars per token)"""
    return len(text) // 4


def render_chat_messages():
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "metadata" in msg:
                model_info = msg['metadata'].get('model', 'Unknown')
                latency = msg['metadata'].get('latency', 'N/A')
                st.caption(f"🚀 {model_info} | ⏱️ {latency} | 🕒 {msg['metadata']['time']}")
                
                if msg.get("role") == "assistant" and msg["metadata"].get("sources_text"):
                    # Render nice badge instead of raw text
                    with st.expander("📝 View Citations & References", expanded=False):
                        st.markdown(msg["metadata"]["sources_text"])


def handle_user_input(chain, model_provider, model_name):
    # Auto-Summary Trigger
    if st.session_state.get("auto_summary_requested"):
        st.session_state.auto_summary_requested = False
        execute_ai_action(chain, model_provider, model_name, "Briefly summarize these documents and list 3 key takeaways.")

    # Quick Action Buttons
    if st.session_state.processed:
        cols = st.columns([1, 1, 1, 2])
        if cols[0].button("📝 Summary"):
            execute_ai_action(chain, model_provider, model_name, "Provide a comprehensive summary of these documents.")
        if cols[1].button("💡 Ideas"):
            execute_ai_action(chain, model_provider, model_name, "Suggest 3 interesting questions I can ask about these files.")
        if cols[2].button("🛠️ Actions"):
            execute_ai_action(chain, model_provider, model_name, "What are the core action items or conclusions in these files?")

    if prompt := st.chat_input("Ask about your documents..."):
        execute_ai_action(chain, model_provider, model_name, prompt)

def execute_ai_action(chain, model_provider, model_name, prompt):
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        import time
        start_time = time.time()
        sources_list = []
        
        # Streaming Generator for GPT-feel
        def response_generator():
            full_response = ""
            # For Parallel chains, we get chunks of the dictionary
            for chunk in chain.stream(prompt):
                if "answer" in chunk:
                    content = chunk["answer"]
                    full_response += content
                    yield content
                if "sources" in chunk:
                    sources_list.extend(chunk["sources"])
            
            st.session_state._last_response = full_response

        # Display streaming response
        full_answer = st.write_stream(response_generator())
        latency = time.time() - start_time
        
        # Display Citations if sources found
        if sources_list:
            unique_sources = {}
            for doc in sources_list:
                src = doc.metadata.get("source", "Unknown")
                pg = doc.metadata.get("page", "?")
                if src not in unique_sources:
                    unique_sources[src] = set()
                unique_sources[src].add(str(pg))
            
            citation_content = " | ".join([f"**{k}** (Pg. {', '.join(sorted(v))})" for k, v in unique_sources.items()])
            citation_str = f"📄 **Sources**: {citation_content}"
            
            # Interactive Badge UI
            st.markdown(f"""
                <div style='background: rgba(0, 212, 255, 0.05); border-left: 4px solid #00d4ff; padding: 12px; border-radius: 8px; margin-top: 15px;'>
                    <span style='color: #00d4ff; font-weight: bold;'>🔍 VERIFIED SOURCE:</span><br>
                    <span style='font-size: 0.85rem;'>{citation_content}</span>
                </div>
            """, unsafe_allow_html=True)


        # Developer Insights Logic
        if st.session_state.get("dev_mode") and sources_list:
            with st.expander("🔍 Developer Insights: Retrieved Context Chunks"):
                for i, doc in enumerate(sources_list):
                    st.markdown(f"**Chunk {i+1}** (Source: {doc.metadata.get('source', 'Unknown')} | Page: {doc.metadata.get('page', '?')})")
                    st.code(doc.page_content, language="text")
                    st.divider()

        # Finalize metadata and history
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": full_answer,
            "metadata": {
                "model": f"{model_provider}/{model_name}",
                "time": datetime.now().strftime("%H:%M:%S"),
                "latency": f"{latency:.2f}s",
                "sources_text": citation_str if sources_list else ""
            }
        })
        
        # Update Analytics
        st.session_state.analytics["total_queries"] += 1
        st.session_state.analytics["total_tokens"] += estimate_tokens(prompt + full_answer)
        current_avg = st.session_state.analytics["avg_latency"]
        queries = st.session_state.analytics["total_queries"]
        st.session_state.analytics["avg_latency"] = (current_avg * (queries-1) + latency) / queries


def render_download_history():
    if st.session_state.chat_history:
        df = pd.DataFrame([
            {"Role": m["role"], "Content": m["content"], "Time": m.get("metadata", {}).get("time", "")}
            for m in st.session_state.chat_history
        ])
        st.sidebar.download_button(
            "📥 Download History",
            data=df.to_csv(index=False),
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
