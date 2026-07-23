import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables (for local testing)
load_dotenv()

def test_pinecone_index_exists():
    """
    Tests if the Pinecone API key is present, connects to Pinecone,
    verifies that the target index exists, and checks that it contains vectors.
    """
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = "social-textbook-index"
    
    # 1. Assert API key exists
    assert pinecone_api_key is not None and len(pinecone_api_key) > 0, \
        "PINECONE_API_KEY is missing from environment variables."
    
    # 2. Initialize Pinecone client
    pc = Pinecone(api_key=pinecone_api_key)
    
    # 3. Assert index exists in Pinecone
    index_exists = pc.has_index(index_name)
    assert index_exists, f"Pinecone index '{index_name}' does not exist!"
    print(f"Success: Pinecone index '{index_name}' found.")
    
    # 4. Assert index contains uploaded vectors (chunks)
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    print(f"Pinecone Index Stats: {stats}")
    
    total_vectors = stats.get('total_vector_count', 0)
    assert total_vectors > 0, f"Pinecone index '{index_name}' exists, but contains 0 vectors!"
    print(f"Success: Index contains {total_vectors} vectors.")

if __name__ == "__main__":
    test_pinecone_index_exists()