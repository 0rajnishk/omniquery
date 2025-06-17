import os
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def load_vectorstore():
    """Load and merge all FAISS stores under DB_FOLDER."""
    DB_FOLDER = "./vectorstores"
    os.makedirs(DB_FOLDER, exist_ok=True)
    
    stores = []
    for dir_name in os.listdir(DB_FOLDER):
        path = os.path.join(DB_FOLDER, dir_name)
        if os.path.isdir(path):
            try:
                store = FAISS.load_local(
                    path, embeddings, allow_dangerous_deserialization=True
                )
                stores.append(store)
            except Exception as e:
                print(f"Error loading store from {path}: {e}")
    
    if not stores:
        return None
        
    base = stores[0]
    for other in stores[1:]:
        base.merge_from(other)
    return base

def similarity_search(query: str, k: int = 5) -> List[str]:
    """
    Perform similarity search on the vector store.
    
    Args:
        query: The search query
        k: Number of results to return
        
    Returns:
        List of relevant document chunks
    """
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return ["No documents available in the vector store."]
        
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs] 