#!/usr/bin/env python3
"""
Agentic RAG System - Main FastAPI Application
"""

import logging
import sys
import os
import time
from contextlib import asynccontextmanager
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
<<<<<<< HEAD
    HackRXRequest,
    HackRXResponse
)
from services import chroma_service, ingestion_service, hackrx_service
=======
    PolicyQuestionsRequest,
    PolicyQuestionsResponse,
)
from services import chroma_service, ingestion_service
from services.llm_service import llm_service
>>>>>>> 38e8e6febc071cecb176f8298202b16f4fd14a2e
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

<<<<<<< HEAD
=======
# Policy Questions endpoint
@app.post("/policy-questions/", response_model=PolicyQuestionsResponse, tags=["Insurance Policy"])
async def answer_policy_questions(request: PolicyQuestionsRequest) -> PolicyQuestionsResponse:
    """
    Answer specific National Parivar Mediclaim Plus Policy questions.
    
    This endpoint is designed to handle structured policy questions and return
    detailed answers with confidence scores and inference explanations.
    
    Returns:
    - Detailed answers to each question
    - Confidence scores (0.0-1.0) for each answer
    - Inference explanations showing reasoning
    - Overall processing metrics
    """
    try:
        start_time = time.time()
        logger.info(f"Processing {len(request.questions)} policy questions...")
        
        # Validate request
        if not request.questions:
            raise HTTPException(status_code=400, detail="Questions list cannot be empty")
        
        # Get context from vector database if requested
        context_chunks = []
        if request.use_context:
            try:
                # Search for policy-related documents (reduced to 5 for efficiency)
                search_results = await chroma_service.search_similar(
                    query="National Parivar Mediclaim Plus Policy insurance coverage benefits",
                    n_results=5
                )
                context_chunks = [result.content for result in search_results]
                logger.info(f"Retrieved {len(context_chunks)} context chunks from database")
            except Exception as e:
                logger.warning(f"Failed to retrieve context from database: {e}")
        
        # Process questions through LLM service
        result = await llm_service.answer_policy_questions(
            questions=request.questions,
            context_chunks=context_chunks if context_chunks else None
        )
        
        processing_time = time.time() - start_time
        average_confidence = sum(result["confidence_scores"]) / len(result["confidence_scores"])
        
        logger.info(f"Policy questions processed in {processing_time:.2f}s with avg confidence: {average_confidence:.2f}")
        
        return PolicyQuestionsResponse(
            answers=result["answers"],
            confidence_scores=result["confidence_scores"],
            inferences=result["inferences"],
            average_confidence=average_confidence,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Policy questions processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Policy questions processing failed: {str(e)}")

# System status endpoint
>>>>>>> 38e8e6febc071cecb176f8298202b16f4fd14a2e
@app.get("/status/", tags=["System"])
async def get_system_status():
    # ... (code remains the same)
    return {"status": "operational"}


# --- New Hackathon Specific Endpoint ---
@app.post("/hackrx/run", response_model=HackRXResponse, tags=["Hackathon"])
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
