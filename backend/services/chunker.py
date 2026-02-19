"""
chunker.py — Handles loading and splitting of transcript text.

Key LangChain concepts used:
- RecursiveCharacterTextSplitter: Smart text splitter that tries to split
  at natural boundaries (paragraphs → sentences → words → characters)
  in that order, preserving semantic coherence.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_transcript(file_path: str) -> str:
    """
    Load a transcript text file from disk.

    Args:
        file_path: Path to the .txt transcript file

    Returns:
        The full transcript as a string

    Example:
        text = load_transcript("data/transcripts/meeting_2024.txt")
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            raise ValueError("The transcript file is empty.")

        print(f"✅ Loaded transcript: {len(content)} characters")
        return content

    except FileNotFoundError:
        raise FileNotFoundError(f"Transcript file not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading transcript: {str(e)}")


def split_transcript_into_chunks(text: str, source_name: str = "transcript") -> List[Document]:
    """
    Split a long transcript into smaller, overlapping chunks.

    This function uses LangChain's RecursiveCharacterTextSplitter which:
    1. First tries to split on double newlines (paragraphs)
    2. Then single newlines
    3. Then periods/sentences
    4. Then spaces (words)
    5. Finally individual characters (last resort)

    This "recursive" approach keeps semantically related text together.

    Args:
        text: The full transcript text
        source_name: Label for metadata (helpful for debugging)

    Returns:
        List of LangChain Document objects, each representing one chunk

    Each Document has:
        - doc.page_content: The actual chunk text
        - doc.metadata: Dictionary with source, chunk_index, chunk_size
    """

    # Initialize the text splitter
    # separators: ordered list of strings to split on
    # The splitter tries each separator in order until chunks are small enough
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,          # Max characters per chunk
        chunk_overlap=CHUNK_OVERLAP,    # Characters shared between consecutive chunks
        length_function=len,            # How to measure length (character count)
        separators=["\n\n", "\n", ". ", " ", ""],  # Priority order for splitting
    )

    # Split the text into chunks
    # Each chunk becomes a LangChain Document object
    raw_chunks = splitter.split_text(text)

    # Wrap each chunk in a Document object with metadata
    # Metadata helps us track where each chunk came from
    documents = []
    for i, chunk in enumerate(raw_chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "source": source_name,
                "chunk_index": i,
                "chunk_size": len(chunk),
                "total_chunks": len(raw_chunks),
            }
        )
        documents.append(doc)

    print(f"✅ Split into {len(documents)} chunks")
    print(f"   Average chunk size: {sum(len(d.page_content) for d in documents) // len(documents)} characters")

    return documents


def process_transcript_file(file_path: str) -> List[Document]:
    """
    Convenience function: Load file + split into chunks in one call.

    Args:
        file_path: Path to transcript file

    Returns:
        List of Document chunks ready for embedding
    """
    # Get just the filename for metadata
    source_name = os.path.basename(file_path)

    # Load the text
    text = load_transcript(file_path)

    # Split into chunks
    documents = split_transcript_into_chunks(text, source_name)

    return documents