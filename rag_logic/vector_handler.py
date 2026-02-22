import os
from rag_logic.config import GOOGLE_API_KEY, MODEL_OPTIONS

def get_embeddings(provider, api_key=None):
    """
    Returns the appropriate embedding model based on the selected provider.
    """
    if provider.lower() == "groq":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    elif provider.lower() == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=api_key or GOOGLE_API_KEY
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def create_vectorstore(documents, provider, api_key, backend_type):
    """
    Creates a vectorstore (FAISS or Chroma) from Document objects.
    """
    embedding = get_embeddings(provider, api_key)
    
    if "faiss" in backend_type.lower():
        from langchain_community.vectorstores import FAISS
        store = FAISS.from_documents(documents, embedding)
        return store
    else:
        from langchain_community.vectorstores import Chroma
        persist_path = f"./data/{provider.lower()}_chroma_db"
        store = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=persist_path
        )
        return store

def load_local_vectorstore(provider, api_key, backend_type):
    """
    Attempts to load an existing vectorstore from disk.
    """
    embedding = get_embeddings(provider, api_key)
    
    if "faiss" in backend_type.lower():
        return None
    else:
        from langchain_community.vectorstores import Chroma
        persist_path = f"./data/{provider.lower()}_chroma_db"
        if os.path.exists(persist_path) and os.listdir(persist_path):
            return Chroma(persist_directory=persist_path, embedding_function=embedding)
        return None
