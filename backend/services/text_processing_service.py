import logging
import hashlib
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from .chromadb_service import chroma_service
from .llm_service import llm_service
from models.schemas import HackRXTextRequest, HackRXTextResponse
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

logger = logging.getLogger(__name__)

class TextProcessingService:
    """Service for processing text documents directly without file uploads"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100"))
        )
        
        # Predefined questions for comprehensive document analysis
        self.predefined_questions = [
            "What is the coverage amount or limit?",
            "What are the main exclusions or restrictions?",
            "What is the deductible amount?",
            "What is the policy period or duration?",
            "What are the claim procedures?",
            "What documents are required for claims?",
            "What is the premium amount or cost?",
            "What are the renewal terms and conditions?",
            "What is the grace period for payments?",
            "What are the cancellation terms?",
            "What are the eligibility criteria?",
            "What are the key benefits or features?",
            "What are the waiting periods?",
            "What is the contact information for support?",
            "What are the important deadlines or dates?"
        ]
    
    def _generate_document_id(self, text_content: str) -> str:
        """Generate a unique document ID based on text content"""
        content_hash = hashlib.sha256(text_content.encode()).hexdigest()[:12]
        return f"direct_text_{content_hash}"
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks for processing"""
        try:
            chunks = self.text_splitter.split_text(text)
            # Filter out very short chunks
            chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 20]
            logger.info(f"Split text into {len(chunks)} meaningful chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting text: {str(e)}")
            # Fallback: simple splitting
            chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    async def _ingest_text_content(self, text_content: str, document_id: str) -> bool:
        """Ingest text content into ChromaDB"""
        try:
            logger.info(f"Starting ingestion of text content (length: {len(text_content)} chars)")
            
            # Clear existing document if it exists
            try:
                await chroma_service.delete_by_filter({"document_id": document_id})
                logger.info(f"Cleared existing content for document ID: {document_id}")
            except Exception as e:
                logger.warning(f"Could not clear existing content: {e}")
            
            # Split text into chunks
            chunks = self._split_text_into_chunks(text_content)
            
            if not chunks:
                logger.warning("No chunks created from text content")
                return False
            
            # Prepare documents for ingestion
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    documents.append(chunk)
                    metadatas.append({
                        "document_id": document_id,
                        "source": "direct_input",
                        "chunk_index": i,
                        "chunk_size": len(chunk),
                        "document_type": "text",
                        "ingestion_timestamp": datetime.now().isoformat()
                    })
                    ids.append(f"{document_id}_chunk_{i}")
            
            if documents:
                # Add documents to collection
                await chroma_service.add_documents(documents, metadatas, ids)
                logger.info(f"Successfully ingested {len(documents)} text chunks")
                return True
            else:
                logger.warning("No valid chunks to ingest")
                return False
                
        except Exception as e:
            logger.error(f"Error ingesting text content: {str(e)}")
            return False
    
    async def _extract_entities_from_text(self, text_content: str) -> Dict[str, Any]:
        """Extract entities from text content"""
        try:
            logger.info("Extracting entities from text content")
            
            # Use a sample of the text for entity extraction (first 2000 chars)
            sample_text = text_content[:2000] if len(text_content) > 2000 else text_content
            entities = await llm_service.extract_entities(sample_text)
            
            logger.info(f"Successfully extracted entities: {list(entities.keys())}")
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return {
                "age": None, "procedure": None, "duration": None, "condition": None,
                "policy_type": None, "coverage_amount": None, "document_type": "text",
                "additional_entities": {}
            }
    
    async def _get_context_for_questions(self, document_id: str, questions: List[str]) -> List[str]:
        """Get relevant context chunks for questions"""
        try:
            # Combine questions for better context retrieval
            combined_query = " ".join(questions)
            
            context_results = await chroma_service.search_similar(
                query=combined_query,
                n_results=10,  # Get more results for comprehensive context
                where_filter={"document_id": document_id}
            )
            
            context_chunks = [result.content for result in context_results]
            logger.info(f"Retrieved {len(context_chunks)} context chunks for questions")
            return context_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    async def process_text_request(self, request: HackRXTextRequest) -> HackRXTextResponse:
        """Process a direct text input request"""
        start_time = time.time()
        
        try:
            input_document = request.input_document.strip()
            custom_questions = request.questions or []
            
            logger.info(f"Processing text document (length: {len(input_document)} chars)")
            
            if not input_document:
                raise ValueError("Input document cannot be empty")
            
            # Generate document ID
            document_id = self._generate_document_id(input_document)
            
            # Step 1: Ingest the document content
            ingestion_success = await self._ingest_text_content(input_document, document_id)
            if not ingestion_success:
                raise ValueError("Failed to ingest document content")
            
            # Step 2: Extract entities
            entities = await self._extract_entities_from_text(input_document)
            
            # Step 3: Process predefined questions
            predefined_context = await self._get_context_for_questions(document_id, self.predefined_questions)
            
            if predefined_context:
                predefined_answers = await llm_service.generate_batch_answers(
                    self.predefined_questions, predefined_context
                )
            else:
                predefined_answers = ["No relevant information found in document."] * len(self.predefined_questions)
            
            # Step 4: Process custom questions if provided
            custom_answers = {}
            if custom_questions:
                custom_context = await self._get_context_for_questions(document_id, custom_questions)
                if custom_context:
                    custom_answer_list = await llm_service.generate_batch_answers(
                        custom_questions, custom_context
                    )
                    custom_answers = {q: a for q, a in zip(custom_questions, custom_answer_list)}
                else:
                    custom_answers = {q: "No relevant information found in document." for q in custom_questions}
            
            # Step 5: Prepare response
            processing_time = time.time() - start_time
            
            predefined_qa = {
                question: answer for question, answer in zip(self.predefined_questions, predefined_answers)
            }
            
            document_summary = {
                "length": len(input_document),
                "word_count": len(input_document.split()),
                "chunks_created": len(self._split_text_into_chunks(input_document)),
                "document_id": document_id,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            response = HackRXTextResponse(
                status="success",
                message="Document processed successfully",
                document_summary=document_summary,
                extracted_entities=entities,
                predefined_qa=predefined_qa,
                custom_qa=custom_answers if custom_answers else None,
                processing_time=processing_time
            )
            
            logger.info(f"Successfully processed text document in {processing_time:.2f} seconds")
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing text request: {str(e)}")
            
            # Return error response
            return HackRXTextResponse(
                status="error",
                message=f"Failed to process document: {str(e)}",
                document_summary={"error": str(e)},
                extracted_entities={},
                predefined_qa={},
                custom_qa=None,
                processing_time=processing_time
            )

# Global service instance
text_processing_service = TextProcessingService()
