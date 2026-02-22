# 🤖 RAG Ultimate | Elite Portfolio Edition
### Developed by Taha Zaman

A high-performance, professional RAG (Retrieval-Augmented Generation) ecosystem designed for technical excellence. This project showcases an end-to-end implementation of modern AI patterns using **LangChain Expression Language (LCEL)** and a premium **Glassmorphism UI**.

---

## 🚀 Elite Features

-   **⚡ High-Speed LCEL Architecture**: Fully migrated to LangChain's modern expression language for optimized, multi-threaded chain execution.
-   **📑 Auto-Summary Snapshot**: Generates an immediate professional 3-point executive summary upon document upload.
-   **⚡ Elite Quick Action Insights**: One-click analysis buttons (Summarize Risks, Extract Data, Professional Brief).
-   **📜 Verified Citation Badges**: Professional source referencing with expandable page-level citations for every response.
-   **📊 Real-time Performance Analytics**: Live tracking of query latency, engine status, and **Token Usage Estimation**.
-   **📂 Knowledge Workspace**: A dedicated management system with vector index statistics and snapshot views.
-   **🎨 Premium Glassmorphism UI**: High-end design with CSS-injected blur effects, neon gradients, and professional typography.
-   **💾 Hybrid Vector Storage**: Seamless switching between **FAISS** (Memory-optimized) and **ChromaDB** (Persistent storage).

---

## 🛠️ Technical Stack

-   **Framework**: Streamlit (Advanced UI Customization)
-   **Engine**: LangChain (LCEL & Modular Design)
-   **Providers**: Groq (Llama 3.3), Google Gemini (2.0 Flash)
-   **Embeddings**: HuggingFace / Gemini 
-   **Storage**: FAISS / ChromaDB

---

## ⚡ Setup & Launch

1. **Install Core Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   Create a `.env` file:
   ```env
   GROQ_API_KEY=your_key
   GOOGLE_API_KEY=your_key
   ```

3. **Launch the Engine**:
   ```bash
   python -m streamlit run app.py
   ```

---

## 🏛️ Project Architecture
The codebase is structured for scalability and professional review:
- `rag_logic/llm_handler.py`: High-performance LCEL chain orchestration.
- `rag_logic/vector_handler.py`: Abstraction layer for hybrid vector backend operations.
- `assets/style.css`: Premium design tokens and animation systems.
- `app.py`: Entry point for the Elite UI.

---
*Professional Portfolio Project by Taha Zaman | 2026*
