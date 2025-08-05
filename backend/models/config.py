import os
from typing import Optional
from pydantic import BaseModel

class Settings(BaseModel):
    # OpenRouter API Configuration
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "deepseek/deepseek-chat-v3-0324:free"
    
    # ChromaDB Configuration
    chroma_persist_directory: str = "./data/chroma_db"
    chroma_collection_name: str = "insurance_documents"
    
    # Embedding Model Configuration
    embedding_model_name: str = "all-MiniLM-L6-v2"
    
    # Document Processing Configuration
    chunk_size: int = 250
    chunk_overlap: int = 50
    max_file_size_mb: int = 50
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Agent Configuration
    max_iterations: int = 10
    temperature: float = 0.1
    
    class Config:
        env_file = "../.env"
        case_sensitive = False

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    return settings