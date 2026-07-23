import os
from langchain_community.document_loaders import PyPDFLoader

def load_pdf_data(file_path="data/10th_social_em.pdf"):
    """Loads a PDF document and returns a list of pages."""
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
        
    print(f"Loading document from {file_path}...")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    print(f"Successfully loaded {len(pages)} pages.")
    return pages