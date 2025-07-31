import logging
import time
from typing import Dict, Any, List, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import BaseMessage
from models.schemas import QueryResponse, EntityExtraction, SearchResult, DecisionType
from agents.entity_extractor import EntityExtractorAgent
from agents.vector_searcher import VectorSearchAgent  
from agents.final_reasoner import FinalReasonerAgent

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State shared between agents"""
    query: str
    max_results: int
    entities: Dict[str, Any]
    search_results: List[Dict[str, Any]]
    final_decision: Dict[str, Any]
    errors: List[str]
    start_time: float
    current_step: str
    entity_extraction_status: str
    search_status: str
    reasoning_status: str
    search_metadata: Dict[str, Any]
    reasoning_metadata: Dict[str, Any]

class AgentCoordinator:
    """Coordinates the multi-agent workflow using LangGraph"""
    
    def __init__(self):
        self.entity_extractor = EntityExtractorAgent()
        self.vector_searcher = VectorSearchAgent()
        self.final_reasoner = FinalReasonerAgent()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Define the workflow steps
        async def extract_entities_step(state: AgentState) -> AgentState:
            """Step 1: Extract entities from query"""
            logger.info("Starting entity extraction step")
            state["current_step"] = "entity_extraction"
            result = await self.entity_extractor.run(state)
            logger.info(f"Entity extraction completed with status: {result.get('entity_extraction_status')}")
            return result
        
        async def vector_search_step(state: AgentState) -> AgentState:
            """Step 2: Perform vector search"""
            logger.info("Starting vector search step")  
            state["current_step"] = "vector_search"
            result = await self.vector_searcher.run(state)
            logger.info(f"Vector search completed with status: {result.get('search_status')}")
            return result
        
        async def final_reasoning_step(state: AgentState) -> AgentState:
            """Step 3: Generate final decision"""
            logger.info("Starting final reasoning step")
            state["current_step"] = "final_reasoning"
            result = await self.final_reasoner.run(state)
            logger.info(f"Final reasoning completed with status: {result.get('reasoning_status')}")
            return result
        
        def should_continue_after_entities(state: AgentState) -> Literal["vector_search", "end"]:
            """Decide whether to continue after entity extraction"""
            if state.get("entity_extraction_status") == "success":
                return "vector_search"
            else:
                logger.warning("Entity extraction failed, ending workflow")
                return "end"
        
        def should_continue_after_search(state: AgentState) -> Literal["final_reasoning", "end"]:
            """Decide whether to continue after vector search"""
            if state.get("search_status") == "success" and state.get("search_results"):
                return "final_reasoning"
            else:
                # Even if search fails, we can still generate a decision with no context
                return "final_reasoning"
        
        # Build the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("extract_entities", extract_entities_step)
        workflow.add_node("vector_search", vector_search_step)  
        workflow.add_node("final_reasoning", final_reasoning_step)
        
        # Add edges
        workflow.set_entry_point("extract_entities")
        workflow.add_conditional_edges(
            "extract_entities",
            should_continue_after_entities,
            {
                "vector_search": "vector_search",
                "end": END
            }
        )
        workflow.add_conditional_edges(
            "vector_search", 
            should_continue_after_search,
            {
                "final_reasoning": "final_reasoning",
                "end": END
            }
        )
        workflow.add_edge("final_reasoning", END)
        
        return workflow.compile()
    
    async def process_query(self, query: str, max_results: int = 5) -> QueryResponse:
        """Process a query through the complete agent workflow"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting query processing: {query[:100]}...")
            
            # Initialize state
            initial_state: AgentState = {
                "query": query,
                "max_results": max_results,
                "entities": {},
                "search_results": [],
                "final_decision": {},
                "errors": [],
                "start_time": start_time,
                "current_step": "initialization",
                "entity_extraction_status": "pending",
                "search_status": "pending", 
                "reasoning_status": "pending",
                "search_metadata": {},
                "reasoning_metadata": {}
            }
            
            # Run the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Extract results from final state
            entities = final_state.get("entities", {})
            search_results = final_state.get("search_results", [])
            decision_data = final_state.get("final_decision", {})
            errors = final_state.get("errors", [])
            
            # Convert entities to EntityExtraction model
            entity_extraction = EntityExtraction(
                age=entities.get("age"),
                procedure=entities.get("procedure"),
                duration=entities.get("duration"),
                condition=entities.get("condition"),
                policy_type=entities.get("policy_type"),
                coverage_amount=entities.get("coverage_amount"),
                additional_entities=entities.get("additional_entities", {})
            )
            
            # Convert search results to SearchResult models
            search_result_objects = []
            for result in search_results:
                search_result_objects.append(SearchResult(
                    chunk_id=result["chunk_id"],
                    content=result["content"],
                    similarity_score=result["similarity_score"],
                    metadata=result.get("metadata", {})
                ))
            
            # Extract decision information
            decision_type = decision_data.get("decision", DecisionType.INSUFFICIENT_INFO.value)
            confidence = decision_data.get("confidence", 0.0)
            reasoning = decision_data.get("reasoning", "No reasoning provided")
            conditions = decision_data.get("conditions")
            citations = decision_data.get("citations", [])
            
            # Validate decision type
            try:
                decision_enum = DecisionType(decision_type)
            except ValueError:
                logger.warning(f"Invalid decision type: {decision_type}, using INSUFFICIENT_INFO")
                decision_enum = DecisionType.INSUFFICIENT_INFO
                confidence = 0.0
            
            # Log any errors
            if errors:
                logger.warning(f"Workflow completed with errors: {errors}")
                if not reasoning.startswith("System error") and decision_enum != DecisionType.INSUFFICIENT_INFO:
                    reasoning += f" (Note: Some processing errors occurred: {'; '.join(errors)})"
            
            # Create response
            response = QueryResponse(
                decision=decision_enum,
                confidence=confidence,
                reasoning=reasoning,
                conditions=conditions,
                citations=citations,
                entities_found=entity_extraction,
                search_results=search_result_objects,
                processing_time=processing_time
            )
            
            logger.info(f"Query processing completed in {processing_time:.2f}s with decision: {decision_enum.value}")
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Query processing failed: {str(e)}")
            
            # Return error response
            return QueryResponse(
                decision=DecisionType.INSUFFICIENT_INFO,
                confidence=0.0,
                reasoning=f"System error occurred during processing: {str(e)}",
                conditions=None,
                citations=[],
                entities_found=EntityExtraction(),
                search_results=[],
                processing_time=processing_time
            )
    
    async def get_workflow_status(self) -> Dict[str, Any]:
        """Get status information about the workflow"""
        try:
            # Get document collection stats
            stats = await self.vector_searcher.get_stats()
            
            return {
                "status": "operational",
                "agents": {
                    "entity_extractor": {
                        "name": self.entity_extractor.name,
                        "description": self.entity_extractor.description
                    },
                    "vector_searcher": {
                        "name": self.vector_searcher.name,
                        "description": self.vector_searcher.description
                    },
                    "final_reasoner": {
                        "name": self.final_reasoner.name,
                        "description": self.final_reasoner.description
                    }
                },
                "document_collection": {
                    "document_count": stats.get("document_count", 0),
                    "collection_name": stats.get("collection_name")
                }
            }
        except Exception as e:
            logger.error(f"Failed to get workflow status: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

# Global coordinator instance
coordinator = AgentCoordinator()
