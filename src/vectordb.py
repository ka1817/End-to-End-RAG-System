import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def get_retriever(docs=None, index_name="social-textbook-index", batch_size=32):
    """
    Connects to Pinecone, creates index if missing, optionally uploads docs, 
    and returns a retriever.
    """
    
    hf_token = os.getenv("HUGGINFACE_ENDPOIN")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    pc = Pinecone(api_key=pinecone_api_key)
    
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2"
    )
    
    if not pc.has_index(index_name):
        print(f"Creating new Pinecone index: '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=768, 
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(5)
    else:
        print(f"Using existing Pinecone index: '{index_name}'.")

    vector_store = PineconeVectorStore(
        index_name=index_name, 
        embedding=embeddings
    )
    
    if docs:
        print("Uploading documents to Pinecone in batches...")
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            print(f"Processing chunks {i} to {i + len(batch)} out of {len(docs)}...")
            
            vector_store.add_documents(documents=batch)
            time.sleep(2)  
            
        print("Upload complete!")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever