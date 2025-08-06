#!/usr/bin/env python3
"""
Agentic RAG System - Main FastAPI Application
"""

import logging
import sys
import os
import time
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import (
    QueryRequest,
    QueryResponse,
    UploadResponse,
    ErrorResponse,
    HackRXRequest,
    HackRXResponse
)
from services import chroma_service, ingestion_service, hackrx_service
from agents import coordinator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agentic_rag.log')
    ]
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting Agentic RAG System...")
    # ... (startup logic remains the same)
    yield
    logger.info("Shutting down Agentic RAG System...")

# Create FastAPI app
app = FastAPI(
    title="Agentic RAG System & Hackathon Endpoint",
    version="1.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Existing Endpoints (Unchanged) ---
@app.get("/health", tags=["System"])
async def health_check():
    # ... (code remains the same)
    return {"status": "healthy"}

@app.post("/upload/", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    # ... (code remains the same)
):
    return {"message": "Upload logic unchanged"}

@app.post("/query/", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest) -> QueryResponse:
    # ... (code remains the same)
    return await coordinator.process_query(query=request.question)

@app.get("/status/", tags=["System"])
async def get_system_status():
    # ... (code remains the same)
    return {"status": "operational"}


# --- New Hackathon Specific Endpoint ---
@app.post("api/v1/hackrx/run", response_model=HackRXResponse, tags=["Hackathon"])
async def run_hackathon_submission(request: HackRXRequest):
    """
    The official endpoint for the HackRx 6.0 challenge.
    Receives a document URL and a list of questions, returns a list of answers.
    """
    logger.info(f"Received /hackrx/run request for document: {request.documents[0]}")
    try:
        response = await hackrx_service.process_request(request)
        return response
    except Exception as e:
        logger.error(f"Critical error in /hackrx/run endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected internal error occurred: {str(e)}"
        )


@app.delete("/documents/{document_url:path}")
async def delete_document_cache(document_url: str):
    """Delete cached document chunks for reprocessing with new parameters"""
    try:
        doc_id = hackrx_service._create_document_id(document_url)
        
        # Delete all chunks for this document
        deleted_count = await chroma_service.delete_by_filter({"document_id": doc_id})
        
        return {
            "message": f"Successfully deleted cached chunks for document",
            "document_id": doc_id[:10] + "...",
            "deleted_chunks": deleted_count
        }
    except Exception as e:
        logger.error(f"Failed to delete document cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete cache: {str(e)}")

@app.delete("/documents/clear-all")
async def clear_all_cache():
    """Clear entire document cache"""
    try:
        deleted_count = await chroma_service.clear_collection()
        return {
            "message": "Successfully cleared all cached documents",
            "deleted_chunks": deleted_count
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@app.get("/chunks/", tags=["ChromaDB"])
async def view_all_chunks(limit: Optional[int] = None):
    """View all chunks stored in ChromaDB"""
    try:
        chunks_data = await chroma_service.get_all_chunks(limit=limit)
        return chunks_data
    except Exception as e:
        logger.error(f"Failed to retrieve chunks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chunks: {str(e)}")


@app.get("/chunks/document/{document_url:path}", tags=["ChromaDB"])
async def view_chunks_by_document(document_url: str, limit: Optional[int] = None):
    """View chunks for a specific document"""
    try:
        doc_id = hackrx_service._create_document_id(document_url)
        chunks_data = await chroma_service.get_chunks_by_document(doc_id, limit=limit)
        
        return {
            "document_url": document_url,
            "document_id": doc_id[:10] + "...",
            **chunks_data
        }
    except Exception as e:
        logger.error(f"Failed to retrieve chunks for document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chunks for document: {str(e)}")


@app.get("/chunks/stats", tags=["ChromaDB"])
async def get_chunks_statistics():
    """Get statistics about chunks in ChromaDB"""
    try:
        total_count = chroma_service.get_collection_count()
        
        # Get sample of chunks to analyze document distribution
        sample_chunks = await chroma_service.get_all_chunks()
        
        # Count documents
        document_ids = set()
        for chunk in sample_chunks.get("chunks", []):
            if "document_id" in chunk.get("metadata", {}):
                document_ids.add(chunk["metadata"]["document_id"])
        
        return {
            "total_chunks": total_count,
            "unique_documents": len(document_ids),
            "collection_name": chroma_service.collection.name if chroma_service.collection else "N/A",
            "sample_size": len(sample_chunks.get("chunks", []))
        }
    except Exception as e:
        logger.error(f"Failed to get chunks statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


# --- Exception Handlers (Unchanged) ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    # ... (code remains the same)
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    # ... (code remains the same)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info"
    )
