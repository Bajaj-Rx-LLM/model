#!/usr/bin/env python3
"""
Agentic RAG System - Main FastAPI Application

A production-ready agentic RAG system for contextual Q&A over 
insurance, legal, and compliance documents using FastAPI, ChromaDB,
sentence-transformers, and OpenRouter LLM API with LangGraph orchestration.
"""

import logging
import sys
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import (
    QueryRequest, 
    QueryResponse, 
    UploadResponse, 
    ErrorResponse,
)
from services import chroma_service, ingestion_service
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

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting Agentic RAG System...")
    
    # Validate configuration
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("OpenRouter API key not configured!")
        raise ValueError("OPENROUTER_API_KEY environment variable must be set")
    
    # Test ChromaDB connection
    try:
        doc_count = chroma_service.get_collection_count()
        logger.info(f"ChromaDB initialized successfully with {doc_count} documents")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {str(e)}")
        raise
    
    # Test agent coordinator
    try:
        status = await coordinator.get_workflow_status()
        logger.info(f"Agent coordinator initialized: {status['status']}")
    except Exception as e:
        logger.error(f"Failed to initialize agent coordinator: {str(e)}")
        raise
    
    logger.info("System startup completed successfully")
    
    yield
    
    logger.info("Shutting down Agentic RAG System...")

# Create FastAPI app
app = FastAPI(
    title="Agentic RAG System",
    description="""
    A production-ready agentic RAG system for contextual question-answering 
    over insurance, legal, and compliance documents.
    
    Features:
    - Multi-format document ingestion (PDF, DOCX, EML, TXT)
    - Semantic vector search with ChromaDB
    - Agent orchestration with LangGraph
    - Structured JSON responses
    - OpenRouter LLM integration
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    try:
        doc_count = chroma_service.get_collection_count()
        status = await coordinator.get_workflow_status()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "services": {
                "chromadb": {
                    "status": "operational",
                    "document_count": doc_count
                },
                "agents": {
                    "status": status.get("status", "unknown")
                }
            },
            "configuration": {
                "model": os.getenv("DEFAULT_MODEL"),
                "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
                "collection_name": os.getenv("CHROMA_COLLECTION_NAME", "documents")
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )

# Document upload endpoint
@app.post("/upload/", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to upload (PDF, DOCX, EML, TXT)")
) -> UploadResponse:
    """
    Upload and process a document for indexing.
    
    Supports PDF, DOCX, EML, and TXT files. The document will be:
    1. Parsed and text extracted
    2. Split into chunks with overlap
    3. Embedded using sentence-transformers
    4. Stored in ChromaDB for retrieval
    
    Returns information about the processed document and number of chunks created.
    """
    try:
        start_time = time.time()
        logger.info(f"Processing upload: {file.filename}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Read file content
        try:
            file_content = await file.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
        
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Process document in background
        async def process_document():
            try:
                # Extract and chunk document
                chunks, metadatas, chunk_ids = await ingestion_service.process_uploaded_file(
                    file_content, file.filename
                )
                
                # Add to vector database
                success = await chroma_service.add_documents(chunks, metadatas, chunk_ids)
                
                if not success:
                    logger.error(f"Failed to add document to ChromaDB: {file.filename}")
                else:
                    logger.info(f"Successfully indexed {file.filename} with {len(chunks)} chunks")
                    
            except Exception as e:
                logger.error(f"Background document processing failed for {file.filename}: {str(e)}")
        
        # Start document processing
        chunks, metadatas, chunk_ids = await ingestion_service.process_uploaded_file(
            file_content, file.filename
        )
        
        # Add to vector database
        success = await chroma_service.add_documents(chunks, metadatas, chunk_ids)
        
        if not success:
            raise HTTPException(
                status_code=500, 
                detail="Document processed but failed to add to vector database"
            )
        
        processing_time = time.time() - start_time
        document_id = metadatas[0]["document_id"] if metadatas else "unknown"
        
        logger.info(f"Upload completed for {file.filename} in {processing_time:.2f}s")
        
        return UploadResponse(
            message=f"Document '{file.filename}' successfully processed and indexed",
            document_id=document_id,
            chunks_processed=len(chunks)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload processing failed for {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

# Query endpoint
@app.post("/query/", response_model=QueryResponse, tags=["Query"])
async def query_documents(request: QueryRequest) -> QueryResponse:
    """
    Query the document collection with natural language.
    
    The system will:
    1. Extract entities from your question
    2. Perform semantic + keyword search over documents
    3. Generate a structured decision with reasoning
    4. Provide citations and confidence scores
    
    Returns a structured JSON response with decision, reasoning, and supporting evidence.
    """
    try:
        start_time = time.time()
        logger.info(f"Processing query: {request.question[:100]}...")
        
        # Validate request
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Process query through agent workflow
        response = await coordinator.process_query(
            query=request.question,
            max_results=request.max_results
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Query processed in {processing_time:.2f}s with decision: {response.decision.value}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

# System status endpoint
@app.get("/status/", tags=["System"])
async def get_system_status():
    """Get detailed system status including agent and database information"""
    try:
        workflow_status = await coordinator.get_workflow_status()
        doc_count = chroma_service.get_collection_count()
        
        return {
            "system": "Agentic RAG System",
            "version": "1.0.0",
            "status": "operational",
            "timestamp": time.time(),
            "components": {
                "vector_database": {
                    "status": "operational",
                    "type": "ChromaDB",
                    "document_count": doc_count,
                    "collection_name": os.getenv("CHROMA_COLLECTION_NAME", "documents"),
                    "embedding_model": os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
                },
                "llm_service": {
                    "status": "operational", 
                    "provider": "OpenRouter",
                    "model": os.getenv("DEFAULT_MODEL"),
                    "base_url": os.getenv("OPENROUTER_BASE_URL")
                },
                "agents": workflow_status.get("agents", {}),
                "document_processing": {
                    "supported_formats": ["PDF", "DOCX", "EML", "TXT"],
                    "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
                    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
                    "max_file_size_mb": int(os.getenv("MAX_FILE_SIZE_MB", "10"))
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "system": "Agentic RAG System",
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
        )

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=str(exc.status_code)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error occurred",
            details=str(exc) if os.getenv("DEBUG", "false").lower() == "true" else None,
            error_code="500"
        ).dict()
    )

if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info"
    )
