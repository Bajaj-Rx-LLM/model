import logging
import os
from typing import Dict, Any, Optional, List
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from models.config import settings
from dotenv import load_dotenv


logger = logging.getLogger(__name__)
load_dotenv()

class LLMService:
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenRouter LLM client"""
        try:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            openrouter_base_url = os.getenv("OPENROUTER_BASE_URL")
            default_model = os.getenv("DEFAULT_MODEL")
            temperature = float(os.getenv("TEMPERATURE", "0.1"))
            
            if not openrouter_api_key:
                raise ValueError("OpenRouter API key not provided")
            
            if not openrouter_base_url:
                raise ValueError("OpenRouter base URL not provided")
            
            logger.info(f"Initializing LLM client with base URL: {openrouter_base_url}")
            logger.info(f"Using model: {default_model}")
            
            # Create ChatOpenAI client configured for OpenRouter
            self.client = ChatOpenAI(
                model=default_model,
                openai_api_key=openrouter_api_key,  # Use older parameter for compatibility
                openai_api_base=openrouter_base_url,  # Use older parameter for compatibility
                temperature=temperature,
                timeout=60,
                max_retries=3,
                streaming=False  # Disable streaming for better compatibility
            )
            
            logger.info(f"LLM client initialized successfully with model: {default_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}")
            raise
    
    async def extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities from user query using LLM"""
        try:
            system_prompt = """You are an expert entity extractor for insurance, legal, and compliance documents.
            
Extract relevant entities from the user's query. Focus on:
- Age (if mentioned)
- Medical procedures or treatments
- Time durations
- Medical conditions or diagnoses
- Policy types (life, health, auto, etc.)
- Coverage amounts or limits
- Any other domain-specific entities

Return your response as a JSON object with the following structure:
{
    "age": "extracted age or null",
    "procedure": "medical procedure or null", 
    "duration": "time period or null",
    "condition": "medical condition or null",
    "policy_type": "insurance policy type or null",
    "coverage_amount": "coverage amount or null",
    "additional_entities": {
        "key": "value"
    }
}

Only extract entities that are explicitly mentioned in the query. Use null for missing information."""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Extract entities from this query: {query}")
            ])
            
            formatted_prompt = prompt.format(query=query)
            response = await asyncio.to_thread(
                self.client.invoke,
                [SystemMessage(content=system_prompt), HumanMessage(content=f"Extract entities from this query: {query}")]
            )
            
            # Parse JSON response
            import json
            try:
                entities = json.loads(response.content)
                logger.info(f"Successfully extracted entities: {entities}")
                return entities
            except json.JSONDecodeError:
                logger.warning("Failed to parse LLM response as JSON, returning empty entities")
                return {
                    "age": None,
                    "procedure": None,
                    "duration": None,
                    "condition": None,
                    "policy_type": None,
                    "coverage_amount": None,
                    "additional_entities": {}
                }
                
        except Exception as e:
            logger.error(f"Failed to extract entities: {str(e)}")
            return {
                "age": None,
                "procedure": None,
                "duration": None,
                "condition": None,
                "policy_type": None,
                "coverage_amount": None,
                "additional_entities": {}
            }
    
    async def generate_final_decision(
        self,
        query: str,
        entities: Dict[str, Any],
        context_chunks: List[str],
        search_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate final decision based on query, entities, and retrieved context"""
        try:
            # Prepare context from search results
            context = "\n\n".join([
                f"Document {i+1} (Similarity: {result.get('similarity_score', 0):.2f}):\n{chunk}"
                for i, (chunk, result) in enumerate(zip(context_chunks, search_results))
            ])
            
            system_prompt = """You are an expert AI assistant for insurance, legal, and compliance document analysis.

Your task is to analyze the user's query against the provided document context and generate a structured decision.

You must respond with a valid JSON object in this exact format:
{
    "decision": "approved|rejected|requires_review|insufficient_info",
    "confidence": 0.85,
    "reasoning": "Detailed explanation of your decision based on the documents",
    "conditions": "Any conditions or requirements (if applicable)",
    "citations": ["doc1_chunk2", "doc2_chunk1"]
}

Decision guidelines:
- "approved": Clear approval based on policy/document criteria
- "rejected": Clear rejection based on policy/document criteria  
- "requires_review": Ambiguous case needing human review
- "insufficient_info": Not enough information in documents to decide

Confidence should be between 0.0 and 1.0.
Citations should reference specific document chunks that support your decision.
Base your decision strictly on the provided context - do not make assumptions beyond what's documented."""
            
            human_prompt = f"""Query: {query}

Extracted Entities: {entities}

Context from Documents:
{context}

Analyze this information and provide your structured decision as JSON."""
            
            response = await asyncio.to_thread(
                self.client.invoke,
                [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            )
            
            # Parse JSON response
            import json
            try:
                decision = json.loads(response.content)
                
                # Validate required fields
                required_fields = ['decision', 'confidence', 'reasoning']
                if not all(field in decision for field in required_fields):
                    raise ValueError("Missing required fields in LLM response")
                
                # Ensure confidence is a float between 0 and 1
                decision['confidence'] = max(0.0, min(1.0, float(decision['confidence'])))
                
                # Ensure citations is a list
                if 'citations' not in decision:
                    decision['citations'] = []
                
                logger.info(f"Generated decision: {decision['decision']} with confidence {decision['confidence']}")
                return decision
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Failed to parse LLM decision response: {e}")
                return {
                    "decision": "insufficient_info",
                    "confidence": 0.1,
                    "reasoning": "Unable to process the response from the AI system. Please try again or contact support.",
                    "conditions": None,
                    "citations": []
                }
                
        except Exception as e:
            logger.error(f"Failed to generate final decision: {str(e)}")
            return {
                "decision": "insufficient_info",
                "confidence": 0.0,
                "reasoning": f"System error occurred during analysis: {str(e)}",
                "conditions": None,
                "citations": []
            }
    
    async def summarize_context(self, context_chunks: List[str], max_length: int = 1000) -> str:
        """Summarize context chunks if they're too long"""
        try:
            combined_context = "\n\n".join(context_chunks)
            
            if len(combined_context) <= max_length:
                return combined_context
            
            system_prompt = f"""Summarize the following document excerpts into a concise overview of no more than {max_length} characters.
Focus on key policies, rules, conditions, and requirements that would be relevant for decision-making."""
            
            response = await asyncio.to_thread(
                self.client.invoke,
                [SystemMessage(content=system_prompt), HumanMessage(content=combined_context)]
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to summarize context: {str(e)}")
            return "\n\n".join(context_chunks)[:max_length]  # Fallback to truncation

# Global LLM service instance
llm_service = LLMService()
