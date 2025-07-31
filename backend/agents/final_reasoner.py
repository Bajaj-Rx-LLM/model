import logging
from typing import Dict, Any, List
from langchain_core.tools import tool
from services.llm_service import llm_service
from models.schemas import DecisionType

logger = logging.getLogger(__name__)

@tool
async def final_reasoning_tool(
    query: str,
    entities: Dict[str, Any],
    search_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate final structured decision based on query, entities, and search results.
    
    Args:
        query: Original user query
        entities: Extracted entities
        search_results: Results from vector search
        
    Returns:
        Dictionary containing the final decision and reasoning
    """
    try:
        logger.info(f"Generating final decision for query: {query[:100]}...")
        
        if not search_results:
            logger.warning("No search results available for reasoning")
            return {
                "status": "success",
                "decision": {
                    "decision": DecisionType.INSUFFICIENT_INFO.value,
                    "confidence": 0.1,
                    "reasoning": "No relevant documents found in the knowledge base to make an informed decision.",
                    "conditions": None,
                    "citations": []
                },
                "tool": "final_reasoner"
            }
        
        # Extract context chunks and result metadata
        context_chunks = [result["content"] for result in search_results]
        result_metadata = [
            {
                "chunk_id": result["chunk_id"],
                "similarity_score": result["similarity_score"],
                "metadata": result.get("metadata", {})
            }
            for result in search_results
        ]
        
        # Generate decision using LLM
        decision = await llm_service.generate_final_decision(
            query=query,
            entities=entities,
            context_chunks=context_chunks,
            search_results=result_metadata
        )
        
        # Validate decision format
        required_fields = ["decision", "confidence", "reasoning"]
        if not all(field in decision for field in required_fields):
            raise ValueError(f"LLM response missing required fields: {required_fields}")
        
        # Ensure decision is valid
        valid_decisions = [d.value for d in DecisionType]
        if decision["decision"] not in valid_decisions:
            logger.warning(f"Invalid decision type: {decision['decision']}, defaulting to insufficient_info")
            decision["decision"] = DecisionType.INSUFFICIENT_INFO.value
        
        # Ensure confidence is in valid range
        decision["confidence"] = max(0.0, min(1.0, float(decision["confidence"])))
        
        # Ensure citations is a list
        if "citations" not in decision:
            decision["citations"] = []
        elif not isinstance(decision["citations"], list):
            decision["citations"] = []
        
        logger.info(f"Generated decision: {decision['decision']} with confidence {decision['confidence']:.2f}")
        
        return {
            "status": "success",
            "decision": decision,
            "context_used": len(context_chunks),
            "tool": "final_reasoner"
        }
        
    except Exception as e:
        logger.error(f"Final reasoning failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "decision": {
                "decision": DecisionType.INSUFFICIENT_INFO.value,
                "confidence": 0.0,
                "reasoning": f"System error occurred during decision generation: {str(e)}",
                "conditions": None,
                "citations": []
            },
            "tool": "final_reasoner"
        }

@tool
async def summarize_context_tool(
    search_results: List[Dict[str, Any]], 
    max_length: int = 1000
) -> Dict[str, Any]:
    """
    Summarize search results context if it's too long.
    
    Args:
        search_results: List of search result dictionaries
        max_length: Maximum length for summary
        
    Returns:
        Dictionary containing summarized context
    """
    try:
        if not search_results:
            return {
                "status": "success",
                "summary": "",
                "original_length": 0,
                "tool": "final_reasoner"
            }
        
        context_chunks = [result["content"] for result in search_results]
        original_text = "\n\n".join(context_chunks)
        original_length = len(original_text)
        
        if original_length <= max_length:
            return {
                "status": "success",
                "summary": original_text,
                "original_length": original_length,
                "summarized": False,
                "tool": "final_reasoner"
            }
        
        # Use LLM service to summarize
        summary = await llm_service.summarize_context(context_chunks, max_length)
        
        return {
            "status": "success",
            "summary": summary,
            "original_length": original_length,
            "summarized": True,
            "compression_ratio": len(summary) / original_length if original_length > 0 else 0,
            "tool": "final_reasoner"
        }
        
    except Exception as e:
        logger.error(f"Context summarization failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "summary": "",
            "original_length": 0,
            "tool": "final_reasoner"
        }

class FinalReasonerAgent:
    """Agent responsible for generating final structured decisions"""
    
    def __init__(self):
        self.name = "final_reasoner"
        self.description = "Generates structured decisions based on retrieved context"
        self.reasoning_tool = final_reasoning_tool
        self.summarize_tool = summarize_context_tool
    
    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute final reasoning"""
        query = state.get("query", "")
        entities = state.get("entities", {})
        search_results = state.get("search_results", [])
        
        # Generate final decision
        reasoning_result = await self.reasoning_tool.ainvoke({
            "query": query,
            "entities": entities,
            "search_results": search_results
        })
        
        # Update state with final decision
        updated_state = state.copy()
        updated_state["final_decision"] = reasoning_result.get("decision", {})
        updated_state["reasoning_status"] = reasoning_result.get("status", "error")
        updated_state["reasoning_metadata"] = {
            "context_used": reasoning_result.get("context_used", 0)
        }
        
        if reasoning_result.get("status") != "success":
            updated_state["errors"] = updated_state.get("errors", [])
            updated_state["errors"].append(f"Final reasoning failed: {reasoning_result.get('error', 'Unknown error')}")
        
        return updated_state
    
    async def summarize_context(self, search_results: List[Dict[str, Any]], max_length: int = 1000) -> Dict[str, Any]:
        """Summarize context if needed"""
        return await self.summarize_tool.ainvoke({
            "search_results": search_results,
            "max_length": max_length
        })
