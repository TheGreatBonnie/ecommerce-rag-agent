"""
This is the main entry point for the e-commerce RAG agent.
It defines the workflow graph, state management, tools, nodes and edges using LangGraph.
This file orchestrates the entire conversational AI system for product search and recommendations.
"""

import json
from functools import lru_cache
from typing import List, TypedDict
from typing_extensions import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from copilotkit.langchain import copilotkit_emit_state, copilotkit_customize_config
from ecommerce_agent.ecommerce import setup_mongodb, search_products, get_product_recommendation, main as setup_ecommerce

# Step 1: Initialize MongoDB collection and load products at startup
# This ensures the database is ready before any user interactions begin
try:
    setup_ecommerce()  # Call main() to ensure products are loaded
    collection = setup_mongodb()
    print("E-commerce system initialized successfully")
except Exception as e:
    print(f"Error initializing ecommerce system: {e}")
    collection = None

class Log(TypedDict):
    """
    Represents a log of an action performed by the agent.
    This provides real-time feedback to users about what the system is doing.
    
    Attributes:
        message: Human-readable description of the current action
        done: Boolean flag indicating if the action is completed
    """
    message: str
    done: bool

class AgentState(MessagesState):
    """
    Enhanced state for the e-commerce agent with comprehensive tracking.
    This extends the base MessagesState to include detailed progress information
    that shows users what the agent is doing during query processing.
    
    The state tracks everything from query parsing to recommendation generation,
    providing transparency into the AI's decision-making process.
    """
    # Step 1: Basic logging and status tracking
    logs: List[Log] = []  # Initialize empty logs list for real-time updates
    processing_status: str = "idle"  # Track overall processing status
    
    # Step 2: Query processing state tracking
    current_query: str = ""  # The current query being processed
    search_stage: str = ""  # Current search stage (parsing, searching, filtering, recommending)
    progress_percentage: int = 0  # Progress indicator from 0-100%
    
    # Step 3: Filter and search result tracking
    active_filters: dict = {}  # Which filters are currently being applied (price, rating, etc.)
    matched_products_count: int = 0  # How many products matched the initial search
    filtered_products_count: int = 0  # How many products remain after filtering
    
    # Step 4: Performance and history tracking
    processing_time: float = 0.0  # How long the current operation has taken
    search_history: list = []  # Store previous search queries and their results
    error_message: str = ""  # Any error message to display to the user

@tool
@lru_cache(maxsize=100)  # Cache recent search results for performance optimization
def search_ecommerce(query: str) -> str:
    """
    Search for products and get AI-powered recommendations based on the query.
    This is the main tool that interfaces with the e-commerce search system.
    
    Args:
        query: Natural language search query from the user
        Example: 'Find me a good laptop for programming under $1500'
        
    Returns:
        Formatted AI-generated recommendation text with product links
    """
    # Step 1: Validate system availability
    if collection is None:
        return "I apologize, but the product search system is currently unavailable. Please try again later."

    try:
        # Step 2: Initialize state tracking variables for progress monitoring
        AgentState.current_query = query
        AgentState.search_stage = "initializing"
        AgentState.progress_percentage = 5
        AgentState.processing_time = 0.0
        AgentState.error_message = ""
        
        # Step 3: Parse query for filters using natural language processing
        # Reset active filters for new query
        AgentState.active_filters = {}
        
        # Step 4: Extract price constraints (e.g., "under $1500", "below $800")
        if "under" in query.lower() or "below" in query.lower() and "$" in query:
            try:
                price_text = query[query.find("$")+1:]
                price_limit = float(price_text.split()[0].replace(",", ""))
                AgentState.active_filters["price_max"] = price_limit
            except Exception:
                pass  # Silently continue if price parsing fails
                
        # Step 5: Extract minimum price constraints (e.g., "above $500", "over $200")
        if "above" in query.lower() or "over" in query.lower() and "$" in query:
            try:
                price_text = query[query.find("$")+1:]
                price_min = float(price_text.split()[0].replace(",", ""))
                AgentState.active_filters["price_min"] = price_min
            except Exception:
                pass  # Silently continue if price parsing fails
        
        # Step 6: Extract rating requirements (e.g., "4 star products", "5-star rated")
        rating_terms = ["star", "rating", "rated"]
        if any(term in query.lower() for term in rating_terms):
            for i in range(1, 6):  # 1 to 5 stars
                if str(i) in query:
                    AgentState.active_filters["min_rating"] = float(i)
                    break
        
        # Step 7: Start timing the search operation for performance monitoring
        import time
        start_time = time.time()
        
        # Step 8: Update search stage to vector search phase
        AgentState.search_stage = "searching"
        AgentState.progress_percentage = 25
        
        # Step 9: Execute semantic search for relevant products using vector embeddings
        print(f"Searching for products matching: {query}")
        relevant_products = search_products(collection, query)
        
        # Step 10: Update search statistics and progress
        AgentState.matched_products_count = len(relevant_products)
        AgentState.progress_percentage = 50
        AgentState.search_stage = "processing results"
        
        # Step 11: Sanitize products to ensure they're JSON serializable
        # This removes any MongoDB-specific objects that could cause serialization issues
        sanitized_products = []
        for product in relevant_products:
            # Create a clean copy without any ObjectId fields
            clean_product = {
                "id": product.get("id", str(product.get("_id", ""))),
                "name": product.get("name", ""),
                "description": product.get("description", ""),
                "price": product.get("price", 0),
                "category": product.get("category", ""),
                "rating": product.get("rating", 0),
                "inStock": product.get("inStock", False),
                "image": product.get("image", "")
            }
            sanitized_products.append(clean_product)
        
        # Step 12: Update filtered products count (after applying any filters)
        AgentState.filtered_products_count = len(sanitized_products)
        
        # Step 13: Log search timing for performance monitoring
        search_time = time.time() - start_time
        AgentState.processing_time = search_time
        if search_time > 15:
            print(f"Warning: Search operation took {search_time:.2f}s")
        
        # Step 14: Log the number of products found for debugging
        print(f"Found {len(sanitized_products)} relevant products for query: {query}")
        
        # Step 15: Handle case where no products were found
        if not sanitized_products:
            AgentState.error_message = "No products found matching your criteria"
            return "I couldn't find any products matching your criteria. Could you try a different search or broaden your requirements?"
        
        # Step 16: Store search results in search history with the actual products
        # Maintain a rolling history of the last 5 searches for reference
        if len(AgentState.search_history) >= 5:  # Keep last 5 searches
            AgentState.search_history.pop(0)  # Remove oldest search
        AgentState.search_history.append({
            "query": query,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "result_count": len(sanitized_products),
            "filters": AgentState.active_filters.copy(),
            "products": sanitized_products  # Store products in search history instead
        })
        
        # Step 17: Update search stage to AI recommendation generation
        AgentState.search_stage = "generating recommendations"
        AgentState.progress_percentage = 75
        
        # Step 18: Generate AI-powered product recommendation using GPT
        print(f"Generating recommendation for query: {query}")
        
        # Generate product recommendation without timeout constraints
        recommendation = get_product_recommendation(query, sanitized_products)
        
        # Step 19: Check if response is too large and truncate if needed
        # This prevents streaming issues and ensures good user experience
        if len(recommendation) > 10000:
            print(f"Warning: Large recommendation generated ({len(recommendation)} chars)")
            # Truncate if necessary to prevent streaming issues
            recommendation = recommendation[:9000] + "\n\n[Note: This recommendation was truncated for better performance]"
        
        # Step 20: Log total time for performance monitoring and update final state
        total_time = time.time() - start_time
        AgentState.processing_time = total_time
        AgentState.progress_percentage = 100
        AgentState.search_stage = "completed"
        
        print(f"Total processing time: {total_time:.2f}s")
        
        return recommendation
    except Exception as e:
        # Step 21: Handle any errors that occur during the search process
        error_message = f"Error during product search: {str(e)}"
        print(error_message)
        import traceback
        print(traceback.format_exc())
        
        # Step 22: Update error state for user feedback
        AgentState.error_message = error_message
        AgentState.search_stage = "error"
        AgentState.progress_percentage = 0
        
        return f"I apologize, but I encountered an error while searching for products. Please try a simpler query or try again later."

# Step 23: Define available tools for the AI agent
# This list contains all tools the agent can use to fulfill user requests
tools = [
    search_ecommerce  # Primary tool for product search and recommendations
]

class ToolNode:
    """
    Node for executing tools in a LangGraph workflow.
    This class handles the execution of tools called by the AI agent,
    manages state updates, and provides real-time feedback to users.
    """
    
    def __init__(self, tools):
        """
        Initialize the ToolNode with available tools.
        
        Args:
            tools: List of tool functions that can be executed
        """
        self.tools = tools
    
    async def __call__(self, state, config):
        """
        Execute tools called by the AI agent with comprehensive state tracking.
        This method handles the entire tool execution lifecycle with real-time updates.
        
        Args:
            state: Current conversation state containing messages and tracking info
            config: Configuration for the execution environment
            
        Returns:
            Updated state with tool execution results
        """
        # Step 1: Initialize or retrieve logs list for user feedback
        state["logs"] = state.get("logs", [])
        
        # Step 2: Update processing status to indicate work is starting
        state["processing_status"] = "processing"
        
        # Step 3: Initialize additional state properties for comprehensive tracking
        state["current_query"] = AgentState.current_query
        state["search_stage"] = AgentState.search_stage
        state["progress_percentage"] = AgentState.progress_percentage
        state["active_filters"] = AgentState.active_filters
        state["matched_products_count"] = AgentState.matched_products_count
        state["filtered_products_count"] = AgentState.filtered_products_count
        state["processing_time"] = AgentState.processing_time
        state["search_history"] = AgentState.search_history
        state["error_message"] = AgentState.error_message
        
        # Step 4: Emit initial state to show we're starting tool execution
        await copilotkit_emit_state(config, state)
        
        # Step 5: Extract tool call information from the last AI message
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or "tool_calls" not in last_message.additional_kwargs:
            return state
            
        tool_calls = last_message.additional_kwargs["tool_calls"]
        
        # Step 6: Execute each tool call requested by the AI
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            tool_call_id = tool_call["id"]
            
            # Step 7: Add a log entry for user visibility of current action
            log_entry = {
                "message": f"Processing {tool_name} with query: {tool_args.get('query', '')}",
                "done": False
            }
            state["logs"].append(log_entry)
            
            # Step 8: Emit state to update frontend about the tool execution starting
            await copilotkit_emit_state(config, state)
            
            # Step 9: Find and execute the matching tool
            for i, tool in enumerate(self.tools):
                if tool.name == tool_name:
                    try:
                        # Step 10: Execute the tool with provided arguments
                        result = await tool.ainvoke(input=tool_args["query"])
                        
                        # Step 11: Add a proper tool message with the tool_call_id
                        tool_message = ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call_id
                        )
                        state["messages"].append(tool_message)
                        
                        # Step 12: Update state with the latest status from AgentState
                        state["current_query"] = AgentState.current_query
                        state["search_stage"] = AgentState.search_stage
                        state["progress_percentage"] = AgentState.progress_percentage
                        state["active_filters"] = AgentState.active_filters
                        state["matched_products_count"] = AgentState.matched_products_count
                        state["filtered_products_count"] = AgentState.filtered_products_count
                        state["processing_time"] = AgentState.processing_time
                        state["search_history"] = AgentState.search_history
                        state["error_message"] = AgentState.error_message
                        
                        # Step 13: Mark the log entry as complete
                        state["logs"][-1]["done"] = True
                        
                        # Step 14: Emit intermediate state after tool execution
                        await copilotkit_emit_state(config, state)
                    except Exception as e:
                        # Step 15: Handle tool execution errors gracefully
                        # Update log with error information
                        state["logs"][-1]["message"] = f"Error in {tool_name}: {str(e)}"
                        state["logs"][-1]["done"] = True
                        
                        # Step 16: Update error state for user feedback
                        state["error_message"] = f"Error in {tool_name}: {str(e)}"
                        state["search_stage"] = "error"
                        state["progress_percentage"] = 0
                        
                        # Step 17: Emit error state to inform user
                        await copilotkit_emit_state(config, state)
                        
                        # Step 18: Add error message to conversation history
                        error_message = ToolMessage(
                            content=f"Error executing {tool_name}: {str(e)}",
                            tool_call_id=tool_call_id
                        )
                        state["messages"].append(error_message)
                    
                    break  # Exit the tool search loop once we found and executed the tool
        
        # Step 19: Clean up and finalize tool execution
        # Clear logs after processing is complete but keep other state data
        state["logs"] = []
        state["processing_status"] = "completed"
        
        # Step 20: Final state emission to indicate completion
        await copilotkit_emit_state(config, state)
        
        return state

async def chat_node(state: AgentState, config: RunnableConfig) -> Command[Literal["tool_node", "__end__"]]:
    """
    Standard chat node based on the ReAct design pattern.
    This node processes user messages, decides whether to call tools or return responses,
    and manages the overall conversation flow.
    
    The ReAct pattern (Reasoning + Acting) allows the AI to:
    1. Reason about the user's request
    2. Act by calling appropriate tools
    3. Observe the results
    4. Reason about next steps
    
    Args:
        state: Current conversation state
        config: Runtime configuration
        
    Returns:
        Command indicating next action (call tools or end conversation)
    """
    # Step 1: Initialize state fields if they don't exist
    state["logs"] = state.get("logs", [])
    state["processing_status"] = "thinking"
    
    # Step 2: Initialize additional tracking properties for comprehensive monitoring
    state["current_query"] = state.get("current_query", "")
    state["search_stage"] = state.get("search_stage", "")
    state["progress_percentage"] = state.get("progress_percentage", 0)
    state["active_filters"] = state.get("active_filters", {})
    state["matched_products_count"] = state.get("matched_products_count", 0)
    state["filtered_products_count"] = state.get("filtered_products_count", 0)
    state["processing_time"] = state.get("processing_time", 0.0)
    state["search_history"] = state.get("search_history", [])
    state["error_message"] = state.get("error_message", "")
    
    # Step 3: Reset error message when starting a new interaction
    state["error_message"] = ""
    
    # Step 4: Emit initial state to frontend for real-time updates
    await copilotkit_emit_state(config, state)
    
    # Step 5: Customize config for intermediate state emission
    # This ensures the frontend receives updates during processing
    config = copilotkit_customize_config(
        config,
        emit_intermediate_state=[],  # No specific intermediate states
        emit_messages=True  # Emit message updates
    )
    
    # Step 6: Define the AI model for conversation processing
    model = ChatOpenAI(model="gpt-4-turbo-preview")

    # Step 7: Bind available tools to the model
    # This tells the AI what tools it can use to fulfill user requests
    model_with_tools = model.bind_tools(
        [
            *state.get("copilotkit", {}).get("actions", []),  # CopilotKit actions
            search_ecommerce  # E-commerce search tool
        ],
        parallel_tool_calls=False,  # Execute tools one at a time for better control
    )

    # Step 8: Define the system message that guides AI behavior
    system_message = SystemMessage(
        content="""You are a helpful e-commerce assistant that can help users find products and provide recommendations. 
        You can search for products and provide detailed recommendations based on user queries.
        
        When users ask about products:
        1. Use the search_ecommerce tool to find relevant products
        2. Present the recommendations in a clear, organized way
        3. Always consider any price constraints or specific requirements mentioned
        4. For each product you recommend, include a clickable link in markdown format: [Product Name](/products/id)
        5. Ensure all product names mentioned are linked to their respective product pages
        """
    )

    # Step 9: Run the model to generate a response
    # This processes the conversation history and generates the next response
    response = await model_with_tools.ainvoke([
        system_message,
        *state["messages"],  # Include full conversation history
    ], config)

    # Step 10: Update processing status to indicate thinking is complete
    state["processing_status"] = "completed"
    
    # Step 11: Emit final state to update frontend
    await copilotkit_emit_state(config, state)

    # Step 12: Check for tool calls in the response and handle them
    if isinstance(response, AIMessage) and response.tool_calls:
        # Get copilotkit actions or empty list if not available
        actions = state.get("copilotkit", {}).get("actions", [])

        # Step 13: Check for any non-copilotkit actions in the response
        # If the AI wants to use our search tool, redirect to tool_node
        if not any(
            action.get("name") == response.tool_calls[0].get("name")
            for action in actions
        ):
            return Command(goto="tool_node", update={"messages": response})

    # Step 14: No tools to call, so we can end the conversation
    # Return the AI's response to the user
    return Command(
        goto=END,
        update={
            "messages": response
        }
    )


# ==================== WORKFLOW GRAPH DEFINITION ====================
# This section defines the LangGraph workflow that orchestrates the entire conversation

# Step 1: Define the workflow graph using LangGraph's StateGraph
# This creates a directed graph where nodes are processing steps and edges define transitions
workflow = StateGraph(AgentState)

# Step 2: Add the chat node that handles user messages and AI responses
# This node processes incoming messages and decides whether to call tools or return a response
workflow.add_node("chat_node", chat_node)

# Step 3: Add the tool node that executes tools like product search 
# This node handles the actual execution of tool functions when requested by the AI
workflow.add_node("tool_node", ToolNode(tools=tools))

# Step 4: Define the transition from tool_node back to chat_node
# After a tool is executed, control returns to the chat node to process the results
workflow.add_edge("tool_node", "chat_node")

# Step 5: Set the entry point of the graph to the chat node
# All conversations start at the chat node which processes the initial message
workflow.set_entry_point("chat_node")

# Step 6: Create a memory checkpointer for storing conversation state
# This enables persistence of conversation history between user interactions
memory_checkpointer = MemorySaver()

# Step 7: Compile the workflow graph with the checkpointer
# This creates an executable graph that can process user requests with memory persistence
graph = workflow.compile(checkpointer=memory_checkpointer)

"""
=== E-COMMERCE RAG AGENT ARCHITECTURE SUMMARY ===

This file implements a conversational AI agent for e-commerce product search using the
LangGraph framework. The agent follows the ReAct (Reasoning + Acting) pattern to provide
intelligent product recommendations through natural language conversations.

SYSTEM ARCHITECTURE:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   User Input    │ -> │   Chat Node      │ -> │   Tool Execution    │
│  "Find laptops" │    │ (GPT-4 Reasoning)│    │ (Product Search)    │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                │                          │
                                v                          v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Final Response │ <- │  Response Formatting │ <- │  AI Recommendations │
│   (with links)  │    │   (Markdown + Links) │    │    (GPT Analysis)   │
└─────────────────┘    └──────────────────┘    └─────────────────────┘

KEY COMPONENTS:

1. AGENT STATE MANAGEMENT
   - Comprehensive state tracking with real-time progress updates
   - Query processing stages (initializing -> searching -> recommending -> completed)
   - Filter extraction and application (price, rating, category, stock status)
   - Performance monitoring with timing and error handling
   - Search history for context and user reference

2. LANGGRAPH WORKFLOW
   - State-based conversation flow using directed graph architecture
   - Two main nodes: chat_node (reasoning) and tool_node (action execution)
   - Memory persistence using MemorySaver for conversation continuity
   - Asynchronous processing with real-time state emission

3. REACT PATTERN IMPLEMENTATION
   - Reasoning: GPT-4 analyzes user queries and decides on actions
   - Acting: Executes search_ecommerce tool to find relevant products  
   - Observing: Processes search results and generates recommendations
   - Iterating: Can chain multiple tool calls if needed

4. TOOL SYSTEM
   - search_ecommerce: Primary tool for product search and recommendations
   - LRU caching for performance optimization of repeated queries
   - Comprehensive error handling with graceful fallbacks
   - Natural language query parsing for filters and constraints

5. REAL-TIME STATE EMISSION
   - CopilotKit integration for frontend state synchronization
   - Progress tracking from 0-100% with descriptive stages
   - Error state management with user-friendly messages
   - Live updates during search, filtering, and recommendation generation

WORKFLOW EXECUTION:
1. User sends message to chat_node
2. GPT-4 processes query and decides if tools are needed
3. If tools required, transitions to tool_node
4. tool_node executes search_ecommerce with query parsing
5. Vector search performed in MongoDB with semantic similarity
6. Results filtered based on extracted constraints
7. AI generates natural language recommendations with product links
8. Results returned to user with full conversation context

STATE TRACKING FEATURES:
- current_query: Active search query being processed
- search_stage: Current processing stage for user feedback
- progress_percentage: Visual progress indicator (0-100%)
- active_filters: Applied price/rating/category/stock filters
- matched_products_count: Initial search results count
- filtered_products_count: Results after applying filters
- processing_time: Performance timing for optimization
- search_history: Last 5 searches with results and filters
- error_message: User-friendly error information

PERFORMANCE OPTIMIZATIONS:
- LRU caching for repeated search queries
- Asynchronous processing with non-blocking operations
- Response truncation to prevent streaming issues
- Parallel-safe state management with comprehensive error handling
- Memory-efficient conversation history with checkpointing

INTEGRATION POINTS:
- MongoDB Vector Search: Semantic product discovery
- OpenAI GPT-4: Natural language understanding and generation
- CopilotKit: Real-time frontend state synchronization
- LangGraph: Workflow orchestration and state management

This architecture provides a robust, scalable foundation for conversational
e-commerce applications with advanced RAG capabilities, real-time feedback,
and intelligent product discovery through natural language interactions.
"""
