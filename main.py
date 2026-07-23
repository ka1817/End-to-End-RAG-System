from src.data_ingestion import load_pdf_data
from src.data_preprocessing import get_chunked_docs
from src.vectordb import get_retriever
from src.retrieval import get_rag_chain

def main():
    print("Starting the RAG Pipeline...\n")
    
    
    print("Connecting to Pinecone and setting up the retriever...")
    retriever = get_retriever()  
    
    rag_chain = get_rag_chain(retriever)
    
    print("\n" + "-"*50)
    print("Chatbot is ready!")
    print("-"*50 + "\n")
    
    user_query = "The Himalayan ranges run in the west-east direction in the form of an arch with a distance of about _______"
    print(f"User Question: {user_query}\n")
    
    print("Thinking...\n")
    response = rag_chain.invoke(user_query)
    
    print("AI Response:")
    print(response)

if __name__ == "__main__":
    main()