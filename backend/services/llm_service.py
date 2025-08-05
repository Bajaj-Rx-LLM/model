import logging
import os
from typing import Dict, Any, Optional, List
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenRouter LLM client"""
        try:
            # --- THIS IS THE CRITICAL CHANGE ---
            # Force-load the .env file from the correct path right before using the variables.
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            dotenv_path = os.path.join(project_root, '.env')
            if os.path.exists(dotenv_path):
                load_dotenv(dotenv_path=dotenv_path)
                logger.info(f"Successfully loaded .env file from: {dotenv_path}")
            else:
                logger.warning(f".env file not found at: {dotenv_path}. Relying on system environment variables.")
            # --- END OF CHANGE ---

            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            openrouter_base_url = os.getenv("OPENROUTER_BASE_URL")
            default_model = os.getenv("DEFAULT_MODEL")
            temperature = float(os.getenv("TEMPERATURE", "0.1"))
            
            if not all([openrouter_api_key, openrouter_base_url, default_model]):
                raise ValueError("OpenRouter API key, Base URL, or Default Model not configured. Check your .env file.")
            
            logger.info(f"Initializing LLM client with model: {default_model}")
            
            self.client = ChatOpenAI(
                model=default_model,
                openai_api_key=openrouter_api_key,
                openai_api_base=openrouter_base_url,
                temperature=temperature,
<<<<<<< HEAD
                timeout=90,
                max_retries=2
=======
                timeout=120,  # Increased timeout for policy questions
                max_retries=3,
                streaming=False  # Disable streaming for better compatibility
>>>>>>> 38e8e6febc071cecb176f8298202b16f4fd14a2e
            )
            logger.info("LLM client initialized successfully.")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}")
            self.client = None

    async def generate_simple_answer(self, question: str, context_chunks: List[str]) -> str:
        if not self.client:
            error_msg = "LLM client is not initialized. Please check your .env configuration and server logs."
            logger.error(error_msg)
            return error_msg
        try:
            context = "\n\n".join(context_chunks)
            system_prompt = """You are an expert AI assistant. Your task is to answer the user's question based *only* on the provided document context.
Be concise and directly answer the question. Do not add any extra information or introductory phrases like 'Based on the context...'.
If the context does not contain the answer, state that the information is not available in the document."""
            
            human_prompt = f"""CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"""
            
            response = await asyncio.to_thread(
                self.client.invoke,
                [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            )
            return response.content.strip()
        except Exception as e:
<<<<<<< HEAD
            logger.error(f"Failed to generate simple answer: {str(e)}")
            return f"An error occurred while generating the answer: {e}"
=======
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
    
    async def answer_policy_questions(self, questions: List[str],policy_name: Optional[str] = None, context_chunks: List[str] = None) -> Dict[str, Any]:
        try:
            # Prepare context if available, but limit size to avoid timeouts
            context = ""
            if context_chunks:
                # Limit context to first 5 most relevant chunks and truncate length
                limited_chunks = context_chunks[:5]
                context_parts = []
                for i, chunk in enumerate(limited_chunks):
                    # Truncate each chunk to 500 characters to avoid token limits
                    truncated_chunk = chunk[:500] + "..." if len(chunk) > 500 else chunk
                    context_parts.append(f"Doc {i+1}: {truncated_chunk}")
                context = "\n\n".join(context_parts)
            
#             system_prompt = """You are an expert insurance document analyst specializing in National Parivar Mediclaim Plus Policy.

# Answer each question with:
# 1. Direct, factual answer
# 2. Confidence score (0.0-1.0) 
# 3. Brief inference explanation

# Use policy context if provided. If information unavailable, state clearly and use lower confidence.

# Response format (JSON only):
# {
#     "answers": ["answer1", "answer2", ...],
#     "confidence_scores": [0.92, 0.87, ...], 
#     "inferences": ["reasoning1", "reasoning2", ...]
# }"""

            system_prompt = """
You are an expert AI assistant specializing in analyzing and answering questions from diverse insurance, legal, or compliance documents, including but not limited to health, life, travel, or automobile insurance policies, terms and conditions, government schemes, and related guidelines.

Your job is to answer each user question based **only on the information in the given context**, without hallucinating. If a question cannot be confidently answered from the context, you must:
- Say the information is unavailable or ambiguous
- Give a lower confidence score
- Avoid guessing or making assumptions

Use the following strict response format (JSON only):

{
    "answers": ["direct factual answer for Q1", "answer for Q2", ...],
    "confidence_scores": [0.95, 0.7, ...],  // Between 0.0 and 1.0
    "inferences": ["brief explanation for Q1", "inference for Q2", ...]
}

Guidelines:
- Use concise language
- Prioritize exact phrases from the context if available
- If multiple interpretations exist, recommend human review
- Be cautious with numeric values, eligibility conditions, and exceptions

Always output valid, structured JSON as described above. No extra commentary or markdown.
"""
            
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
            
            human_prompt = f"""Context: {context if context else 'No policy context available - use general knowledge cautiously'}

Questions:
{questions_text}

Provide JSON response with answers, confidence scores, and inferences."""
            
            response = await asyncio.to_thread(
                self.client.invoke,
                [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            )
            
            # Parse JSON response
            import json
            try:
                result = json.loads(response.content)
                
                # Validate structure
                if not all(key in result for key in ['answers', 'confidence_scores', 'inferences']):
                    raise ValueError("Missing required fields in response")
                
                # Ensure all arrays have same length
                num_questions = len(questions)
                if not (len(result['answers']) == len(result['confidence_scores']) == len(result['inferences']) == num_questions):
                    raise ValueError("Mismatched array lengths in response")
                
                # Normalize confidence scores
                result['confidence_scores'] = [max(0.0, min(1.0, float(score))) for score in result['confidence_scores']]
                
                logger.info(f"Successfully answered {num_questions} policy questions")
                return result
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Failed to parse policy question response: {e}")
                # Fallback response
                return {
                    "answers": ["Not specified in policy document." for _ in questions],
                    "confidence_scores": [0.3 for _ in questions],
                    "inferences": ["Unable to parse LLM response properly." for _ in questions]
                }
                
        except Exception as e:
            logger.error(f"Failed to answer policy questions: {str(e)}")
            
            # Check if it's a timeout error
            if "timeout" in str(e).lower() or "408" in str(e):
                error_msg = "Request timed out. The policy questions were too complex or the service is busy. Please try again."
                confidence = 0.1
            else:
                error_msg = f"System error occurred during analysis: {str(e)}"
                confidence = 0.0
            
            return {
                "answers": [error_msg for _ in questions],
                "confidence_scores": [confidence for _ in questions],
                "inferences": ["Error occurred while processing request." for _ in questions]
            }
>>>>>>> 38e8e6febc071cecb176f8298202b16f4fd14a2e

# Global LLM service instance
llm_service = LLMService()
