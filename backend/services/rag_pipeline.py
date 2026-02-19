"""
rag_pipeline.py — Orchestrates the complete RAG pipeline.

This is the main engine of the application. It coordinates:
1. Document processing (chunking)
2. Vector store creation (embedding + FAISS storage)
3. Retrieval (finding relevant chunks)
4. Generation (LLM creates meeting minutes)

We implement this as a class to maintain state between calls
(the vector store and LLM persist across multiple requests).
"""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Dict, Optional
import os
import sys
import time


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import FAISS_INDEX_DIR

from services.chunker import process_transcript_file, split_transcript_into_chunks
from services.embedder import get_embeddings_model
from services.vector_store import create_vector_store, load_vector_store, FAISS_INDEX_PATH
from services.retriever import create_retriever, retrieve_context
from services.summarizer import load_summarization_model, generate_summary
from config import MAX_CONTEXT_LENGTH

class MeetingMinutesRAGPipeline:
    """
    Complete RAG pipeline for generating meeting minutes.

    Architecture:
    - Maintains a single vector store (updated on each upload)
    - Keeps LLM loaded in memory for fast inference
    - Provides clean interface for the FastAPI router

    Usage:
        pipeline = MeetingMinutesRAGPipeline()
        pipeline.initialize()  # Load models
        result = pipeline.process_transcript("path/to/transcript.txt")
        print(result["minutes"])
    """

    def __init__(self):
        self.vector_store: Optional[FAISS] = None
        self.llm = None
        self.retriever = None
        self._is_initialized = False

    def initialize(self):
        """
        Pre-load all models into memory.

        Call this at startup so the first API request isn't slow.
        Model loading happens once; subsequent requests are fast.
        """
        if self._is_initialized:
            return

        print("🚀 Initializing RAG Pipeline...")
        print("=" * 50)

        # Load embedding model
        print("Step 1/2: Loading embedding model...")
        self.embedding_model = get_embeddings_model()


        # Load summarization LLM
        print("Step 2/2: Loading summarization model...")
        self.llm = load_summarization_model()

        # Try to load existing vector store from disk
        self.vector_store = load_vector_store()
        if self.vector_store:
            self.retriever = create_retriever(self.vector_store)

        self._is_initialized = True
        print("=" * 50)
        print("✅ RAG Pipeline initialized and ready!")

    def process_transcript(self, file_path: str) -> Dict:
        """
        Full pipeline: transcript file → structured meeting minutes.

        This is the main method called by the API endpoint.

        Args:
            file_path: Path to the uploaded transcript .txt file

        Returns:
            Dictionary containing:
            - minutes: Generated meeting minutes text
            - chunks_created: Number of document chunks
            - retrieval_query: The query used for retrieval
            - status: "success" or "error"
        """
        start_time = time.time()

        try:
            # ── Step 1: Load and Chunk ──────────────────────────────────────
            print("\n📄 STEP 1: Processing transcript...")
            documents = process_transcript_file(file_path)

            # ── Step 2: Create Vector Store ─────────────────────────────────
            print("\n🗄️  STEP 2: Building vector store...")
            self.vector_store = create_vector_store(documents)


            # ── Step 3: Create Retriever ─────────────────────────────────────
            print("\n🔍 STEP 3: Setting up retriever...")
            self.retriever = create_retriever(self.vector_store)

            # ── Step 4: Retrieve Relevant Context ────────────────────────────
            print("\n📚 STEP 4: Retrieving relevant context...")

            # We use multiple targeted queries to get comprehensive context
            # Different queries retrieve different relevant sections
            retrieval_queries = [
                "key decisions made in the meeting",
                "action items and assigned responsibilities",
                "deadlines and next steps",
                "meeting discussion summary",
                "participants and attendees",
                "feature roadmap and product planning",
                "project timelines and deliverables"
            ]


            # Collect context from multiple queries for comprehensive coverage
            all_contexts = []
            seen_chunks = set()  # Avoid duplicate chunks

            for query in retrieval_queries:
                docs = self.retriever.invoke(query)
                for doc in docs:
                    # Use content as unique key to deduplicate
                    chunk_key = doc.page_content[:100]
                    if chunk_key not in seen_chunks:
                        seen_chunks.add(chunk_key)
                        all_contexts.append(doc.page_content)

            # Combine all unique retrieved chunks
            combined_context = "\n\n---\n\n".join(all_contexts)

            print(f"✅ Retrieved {len(all_contexts)} unique sections")
            print(f"   Total context length: {len(combined_context)} characters")

            if len(combined_context) > MAX_CONTEXT_LENGTH:
                combined_context = combined_context[:MAX_CONTEXT_LENGTH]

            # ── Step 5: Generate Meeting Minutes ─────────────────────────────
            print("\n🤖 STEP 5: Generating meeting minutes with LLM...")

            generation_query = """
                Generate professional meeting minutes using this exact structure:

                Meeting Overview:
                Brief overview of meeting purpose.

                Key Decisions:
                Bullet points of major decisions.

                Action Items:
                List action items with owner and deadline if available.

                Discussion Summary:
                Key discussion points summarized clearly.

                Next Steps:
                Future actions and plans.

                Write clearly and professionally.
                """


            minutes = generate_summary(self.llm, combined_context, generation_query)

            # ── Post-process and structure the output ─────────────────────────
            # If the LLM output is too short or low quality, add structure
            minutes = self._post_process_minutes(minutes, combined_context)

            elapsed = time.time() - start_time
            print(f"\n✅ Complete! Total time: {elapsed:.1f} seconds")

            return {
                "status": "success",
                "minutes": minutes,
                "chunks_created": len(documents),
                "sections_retrieved": len(all_contexts),
                "processing_time_seconds": round(elapsed, 1),
            }

        except Exception as e:
            print(f"❌ Error in RAG pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e),
                "minutes": None,
            }

    def query_transcript(self, query: str) -> Dict:
        """
        Answer a specific question about the processed transcript.

        After a transcript is processed, users can ask specific questions
        like "Who is responsible for the marketing campaign?" or
        "When is the next meeting?"

        Args:
            query: User's question about the transcript

        Returns:
            Dictionary with the answer and source context
        """
        if not self.vector_store or not self.retriever:
            return {
                "status": "error",
                "error": "No transcript has been processed yet. Please upload a transcript first.",
                "answer": None
            }

        try:
            # Retrieve relevant context for this specific question
            context = retrieve_context(self.retriever, query)

            # Generate a targeted answer
            answer = generate_summary(self.llm, context, query)

            return {
                "status": "success",
                "query": query,
                "answer": answer,
                "context_sections": context[:500] + "..." if len(context) > 500 else context
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "answer": None
            }

    def _post_process_minutes(self, generated_text: str, context: str) -> str:
        """
        Post-process the generated minutes to ensure quality and structure.

        If the LLM generated something too short or low quality,
        we fall back to a structured extraction from the context.

        Args:
            generated_text: Text generated by the LLM
            context: Original retrieved context (fallback)

        Returns:
            Polished meeting minutes string
        """
        # If output is substantial, clean and return it
        if len(generated_text.strip()) > 200:
            # Clean up any prompt artifacts
            cleaned = generated_text.strip()
            # Remove any repeated prompt text that might have leaked through
            for prefix in ["Meeting Minutes:", "Task:", "Generate"]:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
            return cleaned

        # Fallback: Return a structured version of the raw context
        # This ensures users always get useful output
        print("⚠️  LLM output too short — using structured context extraction")
        return f"""MEETING MINUTES (Extracted from Transcript)

{context}

---
Note: The above sections were automatically extracted from the transcript as the most relevant content.
"""


# ─── Singleton Instance ───────────────────────────────────────────────────────
# Create one instance shared across the entire application
# This ensures models are loaded only once
rag_pipeline = MeetingMinutesRAGPipeline()