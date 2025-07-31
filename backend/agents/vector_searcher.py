import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from services.chromadb_service import chroma_service
from models.schemas import SearchResult

logger = logging.getLogger(__name__)

@tool
async def vector_search_tool(
    query: str, 
    entities: Dict[str, Any] = None, 
    max_results: int = 5
) -> Dict[str, Any]:
    """
    Perform semantic vector search and optional hybrid search with keyword matching.
    
    Args:
        query: The search query
        entities: Extracted entities for keyword enhancement
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary containing search results
    """
    try:
        logger.info(f"Performing vector search for query: {query[:100]}...")
        
        # Extract keywords from entities for hybrid search
        keywords = []
        if entities:
            for key, value in entities.items():
                if value and isinstance(value, str) and key != "additional_entities":
                    keywords.append(value.lower())
            
            # Add keywords from additional_entities
            if "additional_entities" in entities and isinstance(entities["additional_entities"], dict):
                for key, value in entities["additional_entities"].items():
                    if value and isinstance(value, str):
                        keywords.append(value.lower())
        
        # Perform hybrid search if keywords are available, otherwise semantic search
        if keywords:
            logger.info(f"Performing hybrid search with keywords: {keywords}")
            search_results = await chroma_service.hybrid_search(
                query=query,
                keywords=keywords,
                n_results=max_results,
                semantic_weight=0.7
            )
        else:
            logger.info("Performing semantic search")
            search_results = await chroma_service.search_similar(
                query=query,
                n_results=max_results
            )
        
        # Convert SearchResult objects to dictionaries
        results_data = []
        for result in search_results:
            results_data.append({
                "chunk_id": result.chunk_id,
                "content": result.content,
                "similarity_score": result.similarity_score,
                "metadata": result.metadata
            })
        
        logger.info(f"Found {len(results_data)} relevant documents")
        return {
            "status": "success",
            "results": results_data,
            "total_results": len(results_data),
            "search_type": "hybrid" if keywords else "semantic",
            "keywords_used": keywords,
            "tool": "vector_searcher"
        }
        
    except Exception as e:
        logger.error(f"Vector search failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "results": [],
            "total_results": 0,
            "tool": "vector_searcher"
        }

@tool
async def get_document_stats_tool() -> Dict[str, Any]:
    """
    Get statistics about the document collection.
    
    Returns:
        Dictionary containing collection statistics
    """
    try:
        count = chroma_service.get_collection_count()
        return {
            "status": "success",
            "document_count": count,
            "collection_name": chroma_service.collection.name if chroma_service.collection else None,
            "tool": "vector_searcher"
        }
    except Exception as e:
        logger.error(f"Failed to get document stats: {str(e)}")
        return {
            "status": "error", 
            "error": str(e),
            "document_count": 0,
            "tool": "vector_searcher"
        }

class VectorSearchAgent:
    """Agent responsible for vector search and retrieval"""
    
    def __init__(self):
        self.name = "vector_searcher"
        self.description = "Performs semantic and hybrid search over document collection"
        self.search_tool = vector_search_tool
        self.stats_tool = get_document_stats_tool
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute vector search"""
        query = state.get("query", "")
        entities = state.get("entities", {})
        max_results = state.get("max_results", 5)
        
        # Perform search
        search_result = await self.search_tool.ainvoke({
            "query": query,
            "entities": entities,
            "max_results": max_results
        })
        
        # Update state with search results
        updated_state = state.copy()
        updated_state["search_results"] = search_result.get("results", [])
        updated_state["search_status"] = search_result.get("status", "error")
        updated_state["search_metadata"] = {
            "total_results": search_result.get("total_results", 0),
            "search_type": search_result.get("search_type", "unknown"),
            "keywords_used": search_result.get("keywords_used", [])
        }
        
        if search_result.get("status") != "success":
            updated_state["errors"] = updated_state.get("errors", [])
            updated_state["errors"].append(f"Vector search failed: {search_result.get('error', 'Unknown error')}")
        
        return updated_state
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return await self.stats_tool.ainvoke({})
