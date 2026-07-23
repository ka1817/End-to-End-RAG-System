from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_chunked_docs(pages):
    """Splits pages into smaller chunks for embedding."""
    
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    
    docs = text_splitter.split_documents(pages)
    
    print(f"Successfully split text into {len(docs)} chunks.")
    return docs