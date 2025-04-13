"""
This is the main entry point for the agent.
It defines the workflow graph, state, tools, nodes and edges.
"""

import json
from functools import lru_cache
from typing_extensions import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from copilotkit import CopilotKitState
from copilotkit.langchain import copilotkit_emit_state, copilotkit_customize_config
from ecommerce_agent.ecommerce import setup_mongodb, search_products, get_product_recommendation, main as setup_ecommerce

# Initialize MongoDB collection and load products
try:
    setup_ecommerce()  # Call main() to ensure products are loaded
    collection = setup_mongodb()
except Exception as e:
    print(f"Error initializing ecommerce system: {e}")
    collection = None

class AgentState(CopilotKitState):
    """
    Here we define the state of the agent
    """
    last_search_results: list = []  # Store last search results for context

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
        # Search for relevant products
        relevant_products = search_products(collection, query)
        
        # Store results in agent state for context
        AgentState.last_search_results = relevant_products
        
        # Get AI recommendation based on the found products
        recommendation = get_product_recommendation(query, relevant_products)
        
        return recommendation
    except Exception as e:
        return f"I apologize, but I encountered an error while searching for products: {str(e)}"

tools = [
    search_ecommerce
]

class ToolNode:
    """Node for executing tools in a workflow graph."""
    
    def __init__(self, tools):
        self.tools = tools
    
    async def __call__(self, state, config):
        # Customize config to control message emission
        config = copilotkit_customize_config(config, emit_messages=False)
        
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
            
            # Find and execute the matching tool
            for tool in self.tools:
                if tool.name == tool_name:
                    # Pass arguments as the input parameter dictionary
                    result = await tool.ainvoke(input=tool_args["query"])
                    
                    # Add a proper tool message with the tool_call_id
                    from langchain_core.messages import ToolMessage
                    tool_message = ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call_id
                    )
                    state["messages"].append(tool_message)
                    
                    # Update state with search results if applicable
                    if tool.name == "search_ecommerce":
                        state["last_search_results"] = AgentState.last_search_results
                    
                    break
        
        # Emit state updates to the frontend
        await copilotkit_emit_state(config, {
            "last_search_results": state.get("last_search_results", []),
            "messages": state.get("messages", []),
            "processing_status": "completed"
        })
        
        return state

async def chat_node(state: AgentState, config: RunnableConfig) -> Command[Literal["tool_node", "__end__"]]:
    """
    Standard chat node based on the ReAct design pattern.
    """
    
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
        """
    )

    # 4. Run the model to generate a response
    response = await model_with_tools.ainvoke([
        system_message,
        *state["messages"],
    ], config)

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

# Define the workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("chat_node", chat_node)
workflow.add_node("tool_node", ToolNode(tools=tools))
workflow.add_edge("tool_node", "chat_node")
workflow.set_entry_point("chat_node")

# Compile the workflow graph
graph = workflow.compile()
