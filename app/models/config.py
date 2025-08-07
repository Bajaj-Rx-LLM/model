import os
from pydantic import BaseModel
from dotenv import load_dotenv

# Load .env file from the project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(project_root, '.env'))

class Settings(BaseModel):
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    default_model: str = os.getenv("DEFAULT_MODEL", "deepseek/deepseek-chat-v3-0324:free")
    
    chroma_persist_directory: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "insurance_documents")
    
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    
    chunk_size: int = int(os.getenv("CHUNK_SIZE", 500))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 100))
    
    temperature: float = float(os.getenv("TEMPERATURE", 0.1))

settings = Settings()

def get_settings() -> Settings:
    return settings
