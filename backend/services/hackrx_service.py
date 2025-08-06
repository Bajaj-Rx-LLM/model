import logging
import hashlib
from typing import List
import asyncio

from .chromadb_service import chroma_service
from .ingestion_service import ingestion_service
from .llm_service import llm_service
from models.schemas import HackRXRequest, HackRXResponse

logger = logging.getLogger(__name__)

class HackRxService:
    def _create_document_id(self, url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()

    async def _get_answer_for_question(self, doc_id: str, question: str) -> str:
        logger.info(f"Searching for context for question: '{question[:50]}...' in doc: '{doc_id[:10]}...'")
        context_results = await chroma_service.search_similar(
            query=question,
            n_results=3,
            where_filter={"document_id": doc_id}
        )
        context_chunks = [result.content for result in context_results]
        if not context_chunks:
            logger.warning("No relevant context found in ChromaDB for the question.")
            return "Could not find relevant information in the provided document to answer the question."
        answer = await llm_service.generate_simple_answer(question, context_chunks)
        return answer

    async def process_request(self, request: HackRXRequest) -> HackRXResponse:
        doc_url = request.documents[0]
        questions = request.questions
        doc_id = self._create_document_id(doc_url)
        logger.info(f"Processing request for document URL: {doc_url} (ID: {doc_id[:10]}...)")

        existing_chunks = await chroma_service.search_similar(query=".", n_results=1, where_filter={"document_id": doc_id})

        if not existing_chunks:
            logger.info(f"Cache MISS for document {doc_id[:10]}.... Processing from URL.")
            try:
                # Run the synchronous ingestion function in a separate thread
                # to avoid blocking the main async application.
                chunks, metadatas, chunk_ids = await asyncio.to_thread(
                    ingestion_service.process_document_from_url, doc_url
                )
                
                # Add the document_id to every chunk's metadata
                for meta in metadatas:
                    meta['document_id'] = doc_id
                
                await chroma_service.add_documents(chunks, metadatas, chunk_ids)
            except Exception as e:
                logger.error(f"Failed to process and ingest document from URL {doc_url}: {e}")
                answers = [f"Error: Could not process the document from the provided URL. Details: {e}" for _ in questions]
                return HackRXResponse(answers=answers)
        else:
            logger.info(f"Cache HIT for document {doc_id[:10]}.... Skipping ingestion.")

        # Get context for all questions (they use the same document)
        logger.info(f"Searching for context for {len(questions)} questions in doc: '{doc_id[:10]}...'")
        context_results = await chroma_service.search_similar(
            query=" ".join(questions),  # Combine all questions for better context retrieval
            n_results=10,  # Get more results since we have multiple questions
            where_filter={"document_id": doc_id}
        )
        context_chunks = [result.content for result in context_results]
        
        if not context_chunks:
            logger.warning("No relevant context found in ChromaDB for the questions.")
            final_answers = ["Could not find relevant information in the provided document to answer the question."] * len(questions)
        else:
            # Answer all questions in a single batch API call
            final_answers = await llm_service.generate_batch_answers(questions, context_chunks)

        return HackRXResponse(answers=final_answers)

hackrx_service = HackRxService()
