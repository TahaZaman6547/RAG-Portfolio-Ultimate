from langchain_core.documents import Document
from rag_logic.pdf_handler import get_text_chunks

def test_metadata_preservation():
    # Mock Document
    doc = Document(
        page_content="This is a test document for RAG verification.",
        metadata={"source": "test.pdf", "page": 1}
    )
    
    # Split into chunks
    chunks = get_text_chunks([doc], chunk_size=10, chunk_overlap=2)
    
    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} Metadata: {chunk.metadata}")
        assert "source" in chunk.metadata
        assert "page" in chunk.metadata
        assert chunk.metadata["source"] == "test.pdf"

if __name__ == "__main__":
    try:
        test_metadata_preservation()
        print("✅ Metadata preservation test PASSED")
    except Exception as e:
        print(f"❌ Test FAILED: {e}")
