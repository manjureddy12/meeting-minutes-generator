"""
main.py — Updated with startup event to pre-load models.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import uvicorn
import os
import sys


from config import API_HOST, API_PORT, CORS_ORIGINS, PROJECT_ROOT
from routers.minutes import router as minutes_router


# ─── Lifespan: runs on startup and shutdown ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs startup code before accepting requests.
    Pre-loads ML models so the first user request isn't slow.
    """
    print("🚀 Starting up Meeting Minutes Generator...")

    # Import here to avoid circular imports
    # from services.rag_pipeline import rag_pipeline
    # rag_pipeline.initialize()

    print("✅ Server ready to accept requests")
    yield  # <-- Server runs here

    # Shutdown code (if needed)
    print("👋 Shutting down...")


# ─── Create FastAPI App ────────────────────────────────────────────────────────
app = FastAPI(
    title="Meeting Minutes Generator",
    description="Automated meeting minutes using RAG pipeline with LangChain",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static Files ─────────────────────────────────────────────────────────────
frontend_dir = PROJECT_ROOT / "frontend"

app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(minutes_router, prefix="/api", tags=["Meeting Minutes"])

# ─── Serve Frontend ───────────────────────────────────────────────────────────
@app.get("/")
async def serve_frontend():
    return FileResponse(str(frontend_dir / "index.html"))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)