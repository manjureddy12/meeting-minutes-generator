"""
config.py — Central configuration for the entire application.

Why have a separate config file?
Because you might need to change model names, paths, or settings
in the future. Having one place to change them is much easier
than hunting through 10 files.
"""

import os
from pathlib import Path

# ─── Project Root Paths ───────────────────────────────────────────────────────
# Path(__file__) = this file's path
# .parent = backend/ folder
# .parent.parent = project root folder
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"

# Create directories if they don't exist
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ─── Model Configuration ──────────────────────────────────────────────────────
# Embedding model: converts text → 384-dimensional vectors
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Summarization model: reads context and generates meeting minutes
# Option 1: google/flan-t5-base (~250MB, faster, good quality)
# Option 2: facebook/bart-large-cnn (~1.6GB, stronger summarization)
SUMMARIZATION_MODEL_NAME = "google/flan-t5-base"

# ─── Text Splitting Configuration ─────────────────────────────────────────────
# CHUNK_SIZE: How many characters per chunk
# Think of it like cutting a long article into paragraphs
CHUNK_SIZE = 500

# CHUNK_OVERLAP: How many characters overlap between consecutive chunks
# This prevents important information from being cut at chunk boundaries
# Example: "...John will lead the Q4 | campaign and report..."
#           Without overlap, "Q4 campaign" would be split. With overlap, it appears in both chunks.
CHUNK_OVERLAP = 50

# ─── Retrieval Configuration ──────────────────────────────────────────────────
# How many chunks to retrieve when answering a question
TOP_K_RESULTS = 5

# Maximum characters passed to LLM to avoid token overflow
MAX_CONTEXT_LENGTH = 12000


# ─── API Configuration ────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = ["*"]  # In production, replace with specific domain