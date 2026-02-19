"""
retriever.py — Creates and manages the LangChain retriever.

The Retriever is the bridge between:
- User's question (text)
- FAISS vector store (vectors)
- Retrieved document chunks (text)

LangChain provides a standard Retriever interface that:
1. Takes a query string
2. Converts it to a vector (using the same embedding model)
3. Searches the vector store
4. Returns relevant Document objects

This standardized interface is important because LangChain chains
(like RetrievalQA) are built to work with any Retriever — so you can
swap FAISS for Pinecone or Chroma without changing anything else.
"""

from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import TOP_K_RESULTS


def create_retriever(vector_store: FAISS) -> BaseRetriever:
    """
    Create a LangChain retriever from a FAISS vector store.

    Args:
        vector_store: Populated FAISS vector store

    Returns:
        LangChain BaseRetriever — the standard retrieval interface

    The retriever is configured with:
    - search_type="similarity": Use cosine similarity to find relevant chunks
    - k: Number of chunks to retrieve per query

    Alternative search types:
    - "mmr" (Maximum Marginal Relevance): Balances similarity + diversity
      Useful when top results are too similar to each other
    - "similarity_score_threshold": Only returns results above a threshold
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",           # Standard cosine similarity search
        search_kwargs={"k": TOP_K_RESULTS}  # Return top 5 most relevant chunks
    )

    print(f"✅ Retriever created (top-{TOP_K_RESULTS} similarity search)")
    return retriever


def retrieve_context(retriever: BaseRetriever, query: str) -> str:
    """
    Use the retriever to fetch relevant chunks and combine them into context.

    This function retrieves relevant document chunks and combines them
    into a single context string that will be passed to the LLM.

    Args:
        retriever: LangChain retriever instance
        query: The question or topic to retrieve context for

    Returns:
        Combined context string from all retrieved chunks

    Example:
        context = retrieve_context(retriever, "What were the action items?")
        # Returns: "1. Sarah Chen will update... 2. Marcus will provide..."
    """
    print(f"🔍 Retrieving context for: '{query}'")

    # get_relevant_documents is the standard LangChain retriever method
    documents: List[Document] = retriever.invoke(query)


    if not documents:
        return "No relevant context found in the transcript."

    # Combine all retrieved chunks into one context string
    # We add the chunk index and a separator for clarity
    context_parts = []
    for i, doc in enumerate(documents):
        context_parts.append(
            f"[Section {i+1}]\n{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    print(f"✅ Retrieved {len(documents)} chunks, total context: {len(context)} characters")
    return context