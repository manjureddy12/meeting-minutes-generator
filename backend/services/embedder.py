"""
embedder.py — Generates vector embeddings from text using HuggingFace models.

Key concept: Embeddings transform text into high-dimensional vectors
where similar meanings cluster together in vector space.

The model we use (all-MiniLM-L6-v2):
- Input: Text string
- Output: 384-dimensional vector (list of 384 floats)
- Size: ~90MB (downloads automatically on first run)
- Speed: Fast enough to run on CPU
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import EMBEDDING_MODEL_NAME


# ─── Global Embeddings Instance ───────────────────────────────────────────────
# We use a global variable so the model is loaded only once.
# Loading an ML model takes time (seconds). If we reload it on every request,
# the API becomes painfully slow.
_embeddings_model = None


def get_embeddings_model() -> HuggingFaceEmbeddings:
    """
    Load and return the HuggingFace embeddings model (singleton pattern).

    The first call downloads the model (~90MB) and loads it into memory.
    Subsequent calls return the already-loaded model instantly.

    Returns:
        HuggingFaceEmbeddings instance ready to convert text to vectors
    """
    global _embeddings_model

    if _embeddings_model is None:
        print(f"⏳ Loading embedding model: {EMBEDDING_MODEL_NAME}")
        print("   (First run downloads ~90MB — this is a one-time operation)")

        _embeddings_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,

            # model_kwargs: passed directly to the underlying HuggingFace model
            model_kwargs={
                "device": "cpu",
                "trust_remote_code": False  # Use CPU (change to "cuda" if you have a GPU)
            },

            # encode_kwargs: controls how text is converted to vectors
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 32  # Normalize vectors to unit length
                # Normalization makes cosine similarity = dot product,
                # which makes comparison faster and more accurate
            }
        )
        print(f"✅ Embedding model loaded successfully")
        print(f"   Embedding dimensions: 384")

    return _embeddings_model


def embed_text(text: str) -> List[float]:
    """
    Convert a single text string to a vector embedding.

    This is used to embed a user's question before searching FAISS.

    Args:
        text: Any text string

    Returns:
        List of 384 floats representing the text's meaning as a vector

    Example:
        vector = embed_text("What were the action items?")
        print(len(vector))  # 384
        print(vector[:3])   # [0.23, -0.45, 0.78]
    """
    model = get_embeddings_model()
    vector = model.embed_query(text)
    return vector


def embed_documents(documents: List[Document]) -> tuple:
    """
    Embed a list of Document chunks.

    Note: In practice, we rarely call this directly — we let LangChain's
    FAISS integration handle embedding + storing in one step.
    This function is useful for debugging and understanding.

    Args:
        documents: List of LangChain Document objects

    Returns:
        Tuple of (documents, list of embedding vectors)
    """
    model = get_embeddings_model()
    texts = [doc.page_content for doc in documents]

    print(f"⏳ Generating embeddings for {len(texts)} chunks...")
    vectors = model.embed_documents(texts)
    print(f"✅ Generated {len(vectors)} embeddings")
    print(f"   Vector shape: each vector has {len(vectors[0])} dimensions")

    return documents, vectors