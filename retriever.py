from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS

def get_retriever(vectorstore: FAISS, strategy: str = "similarity", top_k: int = 3) -> VectorStoreRetriever:
    """
    Returns a configured retriever from the vectorstore.
    
    Strategies:
      "similarity": Uses standard dense retrieval (cosine similarity if normalized).
      "mmr": Uses Max Marginal Relevance to ensure diversity in chunks.
    """
    
    if strategy == "mmr":
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": 20, # Fetch more candidates before selecting top_k for diversity
            }
        )
        print(f"Configured MMR retriever (k={top_k})")
    else:
        # Default to similarity search
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": top_k
            }
        )
        print(f"Configured Similarity retriever (k={top_k})")
        
    return retriever

def retrieve_context(retriever: VectorStoreRetriever, query: str):
    """
    Helper to manually retrieve chunks for debugging/evaluation purposes.
    """
    docs = retriever.invoke(query)
    return docs
