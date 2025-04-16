"""
This is the main entry point for the agent.
It defines the workflow graph, state, tools, nodes and edges.
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

# Initialize MongoDB collection and load products
try:
    setup_ecommerce()  # Call main() to ensure products are loaded
    collection = setup_mongodb()
except Exception as e:
    print(f"Error initializing ecommerce system: {e}")
    collection = None

class Log(TypedDict):
    """
    Represents a log of an action performed by the agent.
    """
    message: str
    done: bool

class AgentState(MessagesState):
    """
    Enhanced state for the ecommerce agent with comprehensive tracking
    to show users what the agent is doing during query processing
    """
    # Existing properties
    logs: List[Log] = []  # Initialize empty logs list
    processing_status: str = "idle"  # Track processing status
    
    # New properties to show query processing state
    current_query: str = ""  # The current query being processed
    search_stage: str = ""  # Current search stage (e.g., "parsing", "searching", "filtering", "recommending")
    progress_percentage: int = 0  # Progress indicator (0-100%)
    active_filters: dict = {}  # Which filters are currently being applied
    matched_products_count: int = 0  # How many products matched the initial search
    filtered_products_count: int = 0  # How many products remain after filtering
    processing_time: float = 0.0  # How long the current operation has taken
    search_history: list = []  # Store previous search queries and their results
    error_message: str = ""  # Any error message to display to the user

@tool
@lru_cache(maxsize=100)  # Cache recent search results
def search_ecommerce(query: str) -> str:
    """
    Search for products and get AI-powered recommendations based on the query.
    Example: 'Find me a good laptop for programming under $1500'
    """
    if collection is None:
        return "I apologize, but the product search system is currently unavailable. Please try again later."

    try:
        # Update state tracking variables
        AgentState.current_query = query
        AgentState.search_stage = "initializing"
        AgentState.progress_percentage = 5
        AgentState.processing_time = 0.0
        AgentState.error_message = ""
        
        # Parse for any filters in the query
        AgentState.active_filters = {}
        if "under" in query.lower() or "below" in query.lower() and "$" in query:
            try:
                price_text = query[query.find("$")+1:]
                price_limit = float(price_text.split()[0].replace(",", ""))
                AgentState.active_filters["price_max"] = price_limit
            except Exception:
                pass
                
        if "above" in query.lower() or "over" in query.lower() and "$" in query:
            try:
                price_text = query[query.find("$")+1:]
                price_min = float(price_text.split()[0].replace(",", ""))
                AgentState.active_filters["price_min"] = price_min
            except Exception:
                pass
        
        # Check if query mentions ratings
        rating_terms = ["star", "rating", "rated"]
        if any(term in query.lower() for term in rating_terms):
            for i in range(1, 6):  # 1 to 5 stars
                if str(i) in query:
                    AgentState.active_filters["min_rating"] = float(i)
                    break
        
        # Add timing for the search operation
        import time
        start_time = time.time()
        
        # Update search stage
        AgentState.search_stage = "searching"
        AgentState.progress_percentage = 25
        
        # Search for relevant products
        print(f"Searching for products matching: {query}")
        relevant_products = search_products(collection, query)
        
        # Update matched products count
        AgentState.matched_products_count = len(relevant_products)
        AgentState.progress_percentage = 50
        AgentState.search_stage = "processing results"
        
        # Sanitize products to ensure they're serializable
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
        
        # Update filtered products count (after applying any filters)
        AgentState.filtered_products_count = len(sanitized_products)
        
        # Log search timing for performance monitoring
        search_time = time.time() - start_time
        AgentState.processing_time = search_time
        if search_time > 15:
            print(f"Warning: Search operation took {search_time:.2f}s")
        
        # Log the number of products found for debugging
        print(f"Found {len(sanitized_products)} relevant products for query: {query}")
        
        # Check if any products were found
        if not sanitized_products:
            AgentState.error_message = "No products found matching your criteria"
            return "I couldn't find any products matching your criteria. Could you try a different search or broaden your requirements?"
        
        # Store search results in search history with the actual products
        if len(AgentState.search_history) >= 5:  # Keep last 5 searches
            AgentState.search_history.pop(0)  # Remove oldest search
        AgentState.search_history.append({
            "query": query,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "result_count": len(sanitized_products),
            "filters": AgentState.active_filters.copy(),
            "products": sanitized_products  # Store products in search history instead
        })
        
        # Update search stage
        AgentState.search_stage = "generating recommendations"
        AgentState.progress_percentage = 75
        
        # Get AI recommendation based on the found products
        print(f"Generating recommendation for query: {query}")
        
        # Generate product recommendation without timeout constraints
        recommendation = get_product_recommendation(query, sanitized_products)
        
        # Check if response is too large (could cause streaming issues)
        if len(recommendation) > 10000:
            print(f"Warning: Large recommendation generated ({len(recommendation)} chars)")
            # Truncate if necessary to prevent streaming issues
            recommendation = recommendation[:9000] + "\n\n[Note: This recommendation was truncated for better performance]"
        
        # Log total time for performance monitoring
        total_time = time.time() - start_time
        AgentState.processing_time = total_time
        AgentState.progress_percentage = 100
        AgentState.search_stage = "completed"
        
        print(f"Total processing time: {total_time:.2f}s")
        
        return recommendation
    except Exception as e:
        error_message = f"Error during product search: {str(e)}"
        print(error_message)
        import traceback
        print(traceback.format_exc())
        
        # Update error state
        AgentState.error_message = error_message
        AgentState.search_stage = "error"
        AgentState.progress_percentage = 0
        
        return f"I apologize, but I encountered an error while searching for products. Please try a simpler query or try again later."

tools = [
    search_ecommerce
]

class ToolNode:
    """Node for executing tools in a workflow graph."""
    
    def __init__(self, tools):
        self.tools = tools
    
    async def __call__(self, state, config):
        # Initialize or retrieve logs list
        state["logs"] = state.get("logs", [])
        
        # Update processing status
        state["processing_status"] = "processing"
        
        # Initialize additional state properties for tracking
        state["current_query"] = AgentState.current_query
        state["search_stage"] = AgentState.search_stage
        state["progress_percentage"] = AgentState.progress_percentage
        state["active_filters"] = AgentState.active_filters
        state["matched_products_count"] = AgentState.matched_products_count
        state["filtered_products_count"] = AgentState.filtered_products_count
        state["processing_time"] = AgentState.processing_time
        state["search_history"] = AgentState.search_history
        state["error_message"] = AgentState.error_message
        
        # Initial state emission to show we're starting
        await copilotkit_emit_state(config, state)
        
        # Get the tool call from the last message if it exists
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or "tool_calls" not in last_message.additional_kwargs:
            return state
            
        tool_calls = last_message.additional_kwargs["tool_calls"]
        
        # Execute each tool call
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            tool_call_id = tool_call["id"]
            
            # Add a log entry for this tool call
            log_entry = {
                "message": f"Processing {tool_name} with query: {tool_args.get('query', '')}",
                "done": False
            }
            state["logs"].append(log_entry)
            
            # Emit state to update frontend about the tool execution starting
            await copilotkit_emit_state(config, state)
            
            # Find and execute the matching tool
            for i, tool in enumerate(self.tools):
                if tool.name == tool_name:
                    try:
                        # Pass arguments as the input parameter dictionary
                        result = await tool.ainvoke(input=tool_args["query"])
                        
                        # Add a proper tool message with the tool_call_id
                        tool_message = ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call_id
                        )
                        state["messages"].append(tool_message)
                        
                        # Update state with the latest status from AgentState
                        state["current_query"] = AgentState.current_query
                        state["search_stage"] = AgentState.search_stage
                        state["progress_percentage"] = AgentState.progress_percentage
                        state["active_filters"] = AgentState.active_filters
                        state["matched_products_count"] = AgentState.matched_products_count
                        state["filtered_products_count"] = AgentState.filtered_products_count
                        state["processing_time"] = AgentState.processing_time
                        state["search_history"] = AgentState.search_history
                        state["error_message"] = AgentState.error_message
                        
                        # Mark the log entry as complete
                        state["logs"][-1]["done"] = True
                        
                        # Emit intermediate state after tool execution
                        await copilotkit_emit_state(config, state)
                    except Exception as e:
                        # Update log with error
                        state["logs"][-1]["message"] = f"Error in {tool_name}: {str(e)}"
                        state["logs"][-1]["done"] = True
                        
                        # Update error state
                        state["error_message"] = f"Error in {tool_name}: {str(e)}"
                        state["search_stage"] = "error"
                        state["progress_percentage"] = 0
                        
                        # Emit error state
                        await copilotkit_emit_state(config, state)
                        
                        # Add error message to messages
                        error_message = ToolMessage(
                            content=f"Error executing {tool_name}: {str(e)}",
                            tool_call_id=tool_call_id
                        )
                        state["messages"].append(error_message)
                    
                    break
        
        # Clear logs after processing is complete but keep other state data
        state["logs"] = []
        state["processing_status"] = "completed"
        
        # Final state emission
        await copilotkit_emit_state(config, state)
        
        return state

async def chat_node(state: AgentState, config: RunnableConfig) -> Command[Literal["tool_node", "__end__"]]:
    """
    Standard chat node based on the ReAct design pattern.
    """
    # Initialize state fields if they don't exist
    state["logs"] = state.get("logs", [])
    state["processing_status"] = "thinking"
    
    # Initialize additional tracking properties
    state["current_query"] = state.get("current_query", "")
    state["search_stage"] = state.get("search_stage", "")
    state["progress_percentage"] = state.get("progress_percentage", 0)
    state["active_filters"] = state.get("active_filters", {})
    state["matched_products_count"] = state.get("matched_products_count", 0)
    state["filtered_products_count"] = state.get("filtered_products_count", 0)
    state["processing_time"] = state.get("processing_time", 0.0)
    state["search_history"] = state.get("search_history", [])
    state["error_message"] = state.get("error_message", "")
    
    # Reset error message when starting a new interaction
    state["error_message"] = ""
    
    # Emit initial state to frontend
    await copilotkit_emit_state(config, state)
    
    # Customize config for intermediate state emission
    config = copilotkit_customize_config(
        config,
        emit_intermediate_state=[],
        emit_messages=True
    )
    
    # 1. Define the model
    model = ChatOpenAI(model="gpt-4-turbo-preview")

    # 2. Bind the tools to the model
    model_with_tools = model.bind_tools(
        [
            *state.get("copilotkit", {}).get("actions", []),
            search_ecommerce
        ],
        parallel_tool_calls=False,
    )

    # 3. Define the system message
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

    # 4. Run the model to generate a response
    response = await model_with_tools.ainvoke([
        system_message,
        *state["messages"],
    ], config)

    # Update processing status
    state["processing_status"] = "completed"
    
    # Final state emission
    await copilotkit_emit_state(config, state)

    # 5. Check for tool calls in the response and handle them
    if isinstance(response, AIMessage) and response.tool_calls:
        # Get copilotkit actions or empty list if not available
        actions = state.get("copilotkit", {}).get("actions", [])

        # 5.1 Check for any non-copilotkit actions in the response
        if not any(
            action.get("name") == response.tool_calls[0].get("name")
            for action in actions
        ):
            return Command(goto="tool_node", update={"messages": response})

    # 6. We've handled all tool calls, so we can end the graph
    return Command(
        goto=END,
        update={
            "messages": response
        }
    )

# Create a memory checkpointer for storing conversation state
memory_checkpointer = MemorySaver()

# Define the workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("chat_node", chat_node)
workflow.add_node("tool_node", ToolNode(tools=tools))
workflow.add_edge("tool_node", "chat_node")
workflow.set_entry_point("chat_node")

# Compile the workflow graph with the checkpointer
graph = workflow.compile(checkpointer=memory_checkpointer)
