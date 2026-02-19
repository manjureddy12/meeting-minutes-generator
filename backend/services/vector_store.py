"""
vector_store.py — Manages FAISS vector database operations.

FAISS (Facebook AI Similarity Search):
- Stores embedding vectors in an efficient index structure
- Can search millions of vectors in milliseconds
- Runs completely locally — no network, no API keys, no cost
- Saves/loads index to/from disk so we don't re-embed on every restart

Flow:
1. Take document chunks
2. Embed each chunk with our embedding model
3. Store vectors in FAISS index
4. When querying: embed the query, find nearest vectors, return their Documents
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Optional
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import FAISS_INDEX_DIR
from services.embedder import get_embeddings_model

# File names for saving/loading the FAISS index to disk
FAISS_INDEX_PATH = str(FAISS_INDEX_DIR / "meeting_index")


def create_vector_store(documents: List[Document]) -> FAISS:
    """
    Create a new FAISS vector store from a list of Document chunks.

    This function:
    1. Gets the embedding model
    2. Embeds all document chunks (converts text → vectors)
    3. Creates FAISS index and stores all vectors
    4. Saves the index to disk for reuse

    Args:
        documents: List of Document objects from the text splitter

    Returns:
        FAISS vector store ready for similarity search

    Note: First run will be slower as it embeds all chunks.
    Subsequent runs load from disk instantly.
    """
    print(f"⏳ Creating FAISS vector store with {len(documents)} chunks...")

    # Get the embedding model
    embeddings = get_embeddings_model()

    # LangChain's FAISS.from_documents() does two things at once:
    # 1. Embeds every document (calls embeddings.embed_documents())
    # 2. Creates FAISS index and inserts all vectors
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )

    # Save the index to disk so we can load it later without re-embedding
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    vector_store.save_local(FAISS_INDEX_PATH)
    print(f"✅ Vector store created and saved to: {FAISS_INDEX_PATH}")
    print(f"   Total vectors stored: {len(documents)}")

    return vector_store


def load_vector_store() -> Optional[FAISS]:
    """
    Load an existing FAISS index from disk.

    Returns:
        FAISS vector store if index exists, None otherwise

    Why load from disk?
    If the server restarts, we don't want to re-embed everything.
    The saved index contains both the vectors AND the original text,
    so we can retrieve full document chunks when searching.
    """
    # Check if saved index exists
    faiss_file = FAISS_INDEX_PATH + ".faiss"
    pkl_file = FAISS_INDEX_PATH + ".pkl"

    if not (os.path.exists(faiss_file) and os.path.exists(pkl_file)):
        print("⚠️ No existing FAISS index found. Will create on first upload.")
        return None


    print(f"⏳ Loading existing FAISS index from disk...")
    embeddings = get_embeddings_model()

    # allow_dangerous_deserialization=True is required because FAISS uses
    # pickle format for serialization. This is safe since we created the file ourselves.
    vector_store = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print(f"✅ FAISS index loaded successfully")
    return vector_store


def similarity_search(vector_store: FAISS, query: str, k: int = 5) -> List[Document]:
    """
    Find the k most relevant document chunks for a given query.

    How it works:
    1. The query text is embedded into a vector
    2. FAISS computes cosine similarity between query vector and all stored vectors
    3. Returns the k documents with highest similarity scores

    Args:
        vector_store: The FAISS instance to search
        query: User's question or topic to find information about
        k: Number of top results to return

    Returns:
        List of most relevant Document chunks

    Example:
        results = similarity_search(vs, "What were the action items?", k=5)
        for doc in results:
            print(doc.page_content)
    """
    print(f"🔍 Searching for: '{query}'")

    # similarity_search_with_score returns (Document, score) pairs
    # Score is the cosine distance (lower = more similar for L2, higher for cosine)
    results_with_scores = vector_store.similarity_search_with_score(query, k=k)

    print(f"✅ Found {len(results_with_scores)} relevant chunks:")
    for i, (doc, score) in enumerate(results_with_scores):
        print(f"   Chunk {i+1}: score={score:.4f} | {doc.page_content[:80]}...")

    # Return just the documents (without scores)
    return [doc for doc, score in results_with_scores]