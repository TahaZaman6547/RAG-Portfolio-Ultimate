from rag_logic.config import GOOGLE_API_KEY, GROQ_API_KEY
from operator import itemgetter

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_llm_chain(model_provider, model, vectorstore, api_key=None):
    """
    Builds and returns a LangChain RAG chain using LCEL.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional research assistant. Answer as detailed as possible using the context below. If you don't find the answer in the context, say 'I don't know.'"),
        ("human", "Context:\n{context}\n\nQuestion:\n{input}")
    ])

    if model_provider.lower() == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=model, api_key=api_key or GROQ_API_KEY)
    elif model_provider.lower() == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=model, api_key=api_key or GOOGLE_API_KEY)
    else:
        raise ValueError(f"Unsupported provider: {model_provider}")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # LCEL Chain Construction with Source Documents
    rag_chain = (
        RunnableParallel({
            "context": retriever | format_docs, 
            "input": RunnablePassthrough()
        })
        | RunnableParallel({
            "answer": prompt | llm | StrOutputParser(),
            "sources": lambda x: retriever.invoke(x["input"])
        })
    )

    return rag_chain
def get_summary_chain(model_provider, model, vectorstore, api_key=None):
    """
    Builds a chain specifically for generating document snapshots.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an Elite Research AI. Generate a concise, professional 3-point bulleted summary (executive snapshot) of the documents provided below. Use professional tone and emojis."),
        ("human", "Documents:\n{context}")
    ])

    if model_provider.lower() == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=model, api_key=api_key or GROQ_API_KEY)
    elif model_provider.lower() == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=model, api_key=api_key or GOOGLE_API_KEY)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Get more context for summary
    
    chain = (
        {"context": retriever | format_docs}
        | prompt 
        | llm 
        | StrOutputParser()
    )
    return chain
