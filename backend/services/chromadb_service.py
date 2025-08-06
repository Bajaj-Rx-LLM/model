import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings as ChromaSettings
from models.config import settings
from models.schemas import SearchResult

# Compatibility fix for some deployment environments
try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

logger = logging.getLogger(__name__)

class ChromaDBService:
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_function = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Ensure the persist directory exists
            os.makedirs(os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db"), exist_ok=True)
            
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db"),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=os.getenv("CHROMA_COLLECTION_NAME", "documents"),
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}  # Use cosine distance
            )
            
            logger.info(f"ChromaDB initialized successfully with collection: {os.getenv('CHROMA_COLLECTION_NAME', 'documents')}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
    async def add_documents(self, chunks: List[str], metadatas: List[Dict[str, Any]], ids: List[str]) -> bool:
        """Add document chunks to the vector database"""
        try:
            # Add documents in batches to avoid memory issues
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                
                self.collection.add(
                    documents=batch_chunks,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            
            logger.info(f"Successfully added {len(chunks)} document chunks to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {str(e)}")
            return False
    
    async def search_similar(
        self, 
        query: str, 
        n_results: int = 10,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar documents"""
        try:
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert results to SearchResult objects
            search_results = []
            if results['ids'] and len(results['ids'][0]) > 0:
                for i in range(len(results['ids'][0])):
                    chunk_id = results['ids'][0][i]
                    content = results['documents'][0][i]
                    distance = results['distances'][0][i]
                    metadata = results['metadatas'][0][i] or {}
                    
                    # Convert distance to similarity score (cosine similarity)
                    similarity_score = 1.0 - distance
                    
                    search_results.append(SearchResult(
                        chunk_id=chunk_id,
                        content=content,
                        similarity_score=similarity_score,
                        metadata=metadata
                    ))
            
            logger.info(f"Found {len(search_results)} similar documents for query")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {str(e)}")
            return []
    
    async def hybrid_search(
        self,
        query: str,
        keywords: List[str],
        n_results: int = 5,
        semantic_weight: float = 0.7
    ) -> List[SearchResult]:
        """Perform hybrid search combining semantic and keyword matching"""
        try:
            # Get semantic search results
            semantic_results = await self.search_similar(query, n_results * 2)
            
            # Get keyword-based results using where_document filter
            keyword_results = []
            if keywords:
                for keyword in keywords:
                    keyword_filter = {"$contains": keyword}
                    kw_results = self.collection.query(
                        query_texts=[query],
                        n_results=n_results,
                        where_document=keyword_filter,
                        include=["documents", "metadatas", "distances"]
                    )
                    
                    if kw_results['ids'] and len(kw_results['ids'][0]) > 0:
                        for i in range(len(kw_results['ids'][0])):
                            chunk_id = kw_results['ids'][0][i]
                            content = kw_results['documents'][0][i]
                            distance = kw_results['distances'][0][i]
                            metadata = kw_results['metadatas'][0][i] or {}
                            
                            similarity_score = 1.0 - distance
                            
                            keyword_results.append(SearchResult(
                                chunk_id=chunk_id,
                                content=content,
                                similarity_score=similarity_score,
                                metadata=metadata
                            ))
            
            # Combine and rank results
            combined_results = {}
            
            # Add semantic results
            for result in semantic_results:
                combined_results[result.chunk_id] = result
                result.similarity_score *= semantic_weight
            
            # Add keyword results
            keyword_weight = 1.0 - semantic_weight
            for result in keyword_results:
                if result.chunk_id in combined_results:
                    # Boost score for documents that match both semantic and keyword search
                    combined_results[result.chunk_id].similarity_score += result.similarity_score * keyword_weight * 0.5
                else:
                    result.similarity_score *= keyword_weight
                    combined_results[result.chunk_id] = result
            
            # Sort by combined score and return top results
            final_results = sorted(
                combined_results.values(),
                key=lambda x: x.similarity_score,
                reverse=True
            )[:n_results]
            
            logger.info(f"Hybrid search returned {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"Failed to perform hybrid search: {str(e)}")
            return await self.search_similar(query, n_results)  # Fallback to semantic search
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection"""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to get collection count: {str(e)}")
            return 0
    
    def reset_collection(self) -> bool:
        """Reset the collection (delete all documents)"""
        try:
            self.client.delete_collection(os.getenv("CHROMA_COLLECTION_NAME", "documents"))
            self.collection = self.client.create_collection(
                name=os.getenv("CHROMA_COLLECTION_NAME", "documents"),
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Collection reset successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reset collection: {str(e)}")
            return False

# Global ChromaDB service instance
chroma_service = ChromaDBService()
