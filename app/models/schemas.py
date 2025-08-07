from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    EML = "eml"
    TXT = "txt"

class UploadResponse(BaseModel):
    message: str
    document_id: str
    chunks_processed: int
    
class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question")
    max_results: int = Field(5, description="Maximum number of similar chunks to retrieve")
    
class EntityExtraction(BaseModel):
    age: Optional[str] = None
    procedure: Optional[str] = None
    duration: Optional[str] = None
    condition: Optional[str] = None
    policy_type: Optional[str] = None
    coverage_amount: Optional[str] = None
    additional_entities: Dict[str, Any] = Field(default_factory=dict)

class SearchResult(BaseModel):
    chunk_id: str
    content: str
    similarity_score: float
    metadata: Dict[str, Any]
    
class DecisionType(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_REVIEW = "requires_review"
    INSUFFICIENT_INFO = "insufficient_info"

class QueryResponse(BaseModel):
    decision: DecisionType
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    conditions: Optional[str] = None
    citations: List[str] = Field(default_factory=list)
    entities_found: EntityExtraction
    search_results: List[SearchResult]
    processing_time: float

class HackRXRequest(BaseModel):
    documents: List[str] = Field(..., description="List of URLs to PDF documents")
    questions: List[str] = Field(..., description="List of questions to answer")

class HackRXResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to the questions")

class PolicyQuestionsRequest(BaseModel):
    questions: List[str] = Field(..., description="List of policy questions")
    use_context: bool = Field(True, description="Whether to use document context")

class PolicyQuestionsResponse(BaseModel):
    answers: List[str]
    confidence_scores: List[float]
    inferences: List[str]
    average_confidence: float
    processing_time: float
    
class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
