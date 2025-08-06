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
                timeout=90,
                max_retries=2
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
            logger.error(f"Failed to generate simple answer: {str(e)}")
            return f"An error occurred while generating the answer: {e}"

    async def generate_batch_answers(self, questions: List[str], context_chunks: List[str]) -> List[str]:
        if not self.client:
            error_msg = "LLM client is not initialized. Please check your .env configuration and server logs."
            logger.error(error_msg)
            return [error_msg] * len(questions)
        try:
            context = "\n\n".join(context_chunks)
            system_prompt = """You are an expert AI assistant. Your task is to answer multiple questions based *only* on the provided document context.
Answer each question concisely and directly. Do not add any extra information or introductory phrases like 'Based on the context...'.
If the context does not contain the answer for a specific question, state that the information is not available in the document for that question.
Format your response as a numbered list with each answer on a new line."""
            
            questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
            human_prompt = f"""CONTEXT:\n{context}\n\nQUESTIONS:\n{questions_text}\n\nANSWERS:"""
            
            response = await asyncio.to_thread(
                self.client.invoke,
                [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            )
            
            # Parse the response to extract individual answers
            response_text = response.content.strip()
            lines = response_text.split('\n')
            answers = []
            
            for line in lines:
                line = line.strip()
                if line and any(line.startswith(f"{i+1}.") for i in range(len(questions))):
                    # Remove the number prefix
                    answer = line.split('.', 1)[1].strip() if '.' in line else line
                    answers.append(answer)
            
            # Ensure we have the correct number of answers
            while len(answers) < len(questions):
                answers.append("Unable to extract answer from response.")
            
            return answers[:len(questions)]
            
        except Exception as e:
            logger.error(f"Failed to generate batch answers: {str(e)}")
            return [f"An error occurred while generating the answer: {e}"] * len(questions)

# Global LLM service instance
llm_service = LLMService()
