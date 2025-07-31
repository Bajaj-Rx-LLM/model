import logging
from typing import Dict, Any
from langchain_core.tools import tool
from services.llm_service import llm_service

logger = logging.getLogger(__name__)

@tool
async def extract_entities_tool(query: str) -> Dict[str, Any]:
    """
    Extract entities from user query for insurance/legal/compliance analysis.
    
    Args:
        query: The user's natural language question
        
    Returns:
        Dictionary containing extracted entities
    """
    try:
        logger.info(f"Extracting entities from query: {query[:100]}...")
        entities = await llm_service.extract_entities(query)
        logger.info(f"Extracted entities: {entities}")
        return {
            "status": "success",
            "entities": entities,
            "tool": "entity_extractor"
        }
    except Exception as e:
        logger.error(f"Entity extraction failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "entities": {
                "age": None,
                "procedure": None,
                "duration": None,
                "condition": None,
                "policy_type": None,
                "coverage_amount": None,
                "additional_entities": {}
            },
            "tool": "entity_extractor"
        }

class EntityExtractorAgent:
    """Agent responsible for extracting entities from user queries"""
    
    def __init__(self):
        self.name = "entity_extractor"
        self.description = "Extracts relevant entities from user queries"
        self.tool = extract_entities_tool
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute entity extraction"""
        query = state.get("query", "")
        result = await self.tool.ainvoke({"query": query})
        
        # Update state with extracted entities
        updated_state = state.copy()
        updated_state["entities"] = result.get("entities", {})
        updated_state["entity_extraction_status"] = result.get("status", "error")
        
        if result.get("status") != "success":
            updated_state["errors"] = updated_state.get("errors", [])
            updated_state["errors"].append(f"Entity extraction failed: {result.get('error', 'Unknown error')}")
        
        return updated_state
