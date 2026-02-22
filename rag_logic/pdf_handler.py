def get_pdf_documents(uploaded_files):
    """
    Extracts text from PDF files and returns a list of LangChain Document objects.
    Preserves metadata (filename and page number) for citations.
    """
    from pypdf import PdfReader
    from langchain_core.documents import Document
    
    documents = []
    for file in uploaded_files:
        try:
            reader = PdfReader(file)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": file.name,
                            "page": i + 1
                        }
                    ))
        except Exception as e:
            print(f"Error reading {file.name}: {e}")
    return documents

def get_text_chunks(documents, chunk_size=3000, chunk_overlap=300):
    """
    Splits Document objects into smaller chunks while preserving metadata.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)
