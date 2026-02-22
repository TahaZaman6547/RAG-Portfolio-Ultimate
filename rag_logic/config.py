import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API keys for different LLM providers
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Dictionary of available model providers and their respective models
MODEL_OPTIONS = {
    "Groq": {
        "playground": "https://console.groq.com/",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
    },
    "Gemini": {
        "playground": "https://ai.google.dev",
        "models": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    }
}

# Vector Storage Options
VECTOR_BACKENDS = ["FAISS (Memory-based)", "ChromaDB (Persistent)"]
