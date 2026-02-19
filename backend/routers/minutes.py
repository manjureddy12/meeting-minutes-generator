"""
minutes.py — FastAPI router for meeting minutes endpoints.

Endpoints:
  POST /api/upload-transcript    — Upload a transcript file → process → return minutes
  POST /api/query                — Ask a question about the processed transcript
  GET  /api/status               — Check pipeline status
  GET  /api/health               — Health check

Why separate router from main.py?
Clean separation of concerns. As the project grows, you might add:
- /api/users (user management)
- /api/history (saved minutes)
- /api/export (PDF export)
Each gets its own router file.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import sys
import shutil

from config import TRANSCRIPTS_DIR
from services.rag_pipeline import rag_pipeline

# Create router
router = APIRouter()


# ─── Pydantic Models (Request/Response Schemas) ───────────────────────────────
# Pydantic models define the structure of request and response JSON
# FastAPI uses these for automatic validation and documentation

class QueryRequest(BaseModel):
    """Schema for the /query endpoint request body."""
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the action items and who is responsible for each?"
            }
        }


class MinutesResponse(BaseModel):
    """Schema for successful minutes generation response."""
    status: str
    minutes: str
    chunks_created: int
    sections_retrieved: int
    processing_time_seconds: float
    filename: str


class QueryResponse(BaseModel):
    """Schema for query response."""
    status: str
    query: str
    answer: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/status")
async def get_status():
    """
    Check if the RAG pipeline is initialized and ready.
    Called by the frontend on page load to show readiness.
    """
    return {
        "status": "ready" if rag_pipeline._is_initialized else "not_initialized",
        "vector_store_loaded": rag_pipeline.vector_store is not None,
        "llm_loaded": rag_pipeline.llm is not None,
        "message": "RAG Pipeline is operational" if rag_pipeline._is_initialized
                   else "Pipeline loading... please wait"
    }


@router.post("/upload-transcript")
async def upload_transcript(file: UploadFile = File(...)):
    """
    Upload a meeting transcript file and generate meeting minutes.

    How file upload works in FastAPI:
    1. Browser sends multipart/form-data POST request with file data
    2. FastAPI receives it as UploadFile object
    3. We save it to disk
    4. We pass the file path to our RAG pipeline
    5. RAG pipeline processes it and returns minutes

    Args:
        file: The uploaded file (must be .txt or .md)

    Returns:
        JSON with generated meeting minutes and metadata
    """
    # ── Validate file type ────────────────────────────────────────────────
    allowed_extensions = {".txt", ".md"}
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file_extension}'. Only .txt and .md files are supported."
        )

    # ── Validate file size (max 5MB) ──────────────────────────────────────
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB in bytes
    content = await file.read()
    await file.close()


    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(content) / 1024 / 1024:.1f}MB). Maximum size is 5MB."
        )

    if len(content) < 100:
        raise HTTPException(
            status_code=400,
            detail="File is too small. Please upload a real meeting transcript (minimum 100 characters)."
        )

    # ── Save file to disk ─────────────────────────────────────────────────
    # Sanitize filename to prevent path traversal attacks
    safe_filename = os.path.basename(file.filename)
    file_path = TRANSCRIPTS_DIR / safe_filename

    with open(file_path, "wb") as f:
        f.write(content)

    print(f"📁 Saved transcript: {file_path} ({len(content)} bytes)")

    # ── Ensure pipeline is initialized ────────────────────────────────────
    if not rag_pipeline._is_initialized:
        rag_pipeline.initialize()

    # ── Run RAG Pipeline ──────────────────────────────────────────────────
    result = rag_pipeline.process_transcript(str(file_path))

    # ── Handle errors ─────────────────────────────────────────────────────
    if result["status"] == "error":
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline error: {result.get('error', 'Unknown error')}"
        )

    # ── Return successful response ────────────────────────────────────────
    return JSONResponse(
        content={
            "status": "success",
            "filename": safe_filename,
            "minutes": result["minutes"],
            "chunks_created": result["chunks_created"],
            "sections_retrieved": result["sections_retrieved"],
            "processing_time_seconds": result["processing_time_seconds"],
        }
    )


@router.post("/query")
async def query_transcript(request: QueryRequest):
    """
    Ask a specific question about the last processed transcript.

    This allows users to drill down into specific aspects:
    - "Who is responsible for the marketing campaign?"
    - "What was decided about the launch date?"
    - "List all deadlines mentioned"

    Args:
        request: QueryRequest with the question

    Returns:
        JSON with the AI-generated answer
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not rag_pipeline._is_initialized:
        rag_pipeline.initialize()

    result = rag_pipeline.query_transcript(request.question)

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result.get("error"))

    return JSONResponse(content=result)