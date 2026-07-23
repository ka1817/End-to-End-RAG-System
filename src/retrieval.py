import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    """Helper function to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(retriever):
    """
    Creates and returns the RAG chain using the provided retriever, 
    a Groq LLM, and LCEL.
    """
    print("Initializing LLM and building the RAG chain...")
    
    # 1. Load API keys for Groq
    load_dotenv()
    os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
    
    # 2. Initialize the LLM using ChatGroq
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile"
    )
    
    # 3. Define the Prompt Template
    template = """You are an expert assistant for answering questions based on textbook material.
Use the following pieces of retrieved context to answer the question.
If you do not know the answer based on the context, say that you don't know.

Context:
{context}

Question: {input}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 4. Construct the LCEL Chain using the pipe operator (|)
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("RAG chain successfully built!")
    return rag_chain