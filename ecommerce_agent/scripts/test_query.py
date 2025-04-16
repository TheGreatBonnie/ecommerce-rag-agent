"""
Script to test the ecommerce agent with sample queries.
Run this script with: poetry run test-query
"""

import asyncio
import argparse
import uuid
from typing import List, Optional
from pprint import pprint
from ecommerce_agent.agent import search_ecommerce, graph

async def test_query(query: str, verbose: bool = False):
    """Test the ecommerce agent with a specific query."""
    print(f"\n🔍 Testing query: \"{query}\"\n")
    
    try:
        # Direct test of search_ecommerce tool
        print("--- Search Results ---")
        result = search_ecommerce(query)
        print(result)
        
        if verbose:
            # Show the raw search results
            from ecommerce_agent.agent import AgentState
            print("\n--- Raw Product Data ---")
            pprint(AgentState.last_search_results)
    except Exception as e:
        print(f"\n❌ Error in search_ecommerce: {e}")
        import traceback
        traceback.print_exc()

async def test_query_with_graph(query: str, verbose: bool = False):
    """Test the ecommerce agent with the full graph workflow."""
    print(f"\n🔍 Testing query with graph: \"{query}\"\n")
    
    try:
        # Use the full graph
        from ecommerce_agent.agent import AgentState
        
        # Initialize a state for the agent
        initial_state = AgentState(
            messages=[{"role": "user", "content": query}]
        )
        
        # Create required config with unique identifiers for the checkpointer
        config = {
            "thread_id": str(uuid.uuid4()),  # Generate a unique thread ID
            "checkpoint_ns": "test_query",   # Namespace for checkpointing
            "checkpoint_id": str(uuid.uuid4())  # Unique checkpoint ID
        }
        
        # Stream the response to show progress
        print("--- Agent Response (streaming) ---")
        response_text = ""
        async for event in graph.astream_events(initial_state, config=config, version="v2"):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    # Print progress character
                    print(".", end="", flush=True)
                    response_text += chunk.content
        
        print("\n\n--- Final Response ---")
        print(response_text)
        
        if verbose:
            # Show the raw search results
            print("\n--- Raw Product Data ---")
            pprint(AgentState.last_search_results)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def run_sample_queries(queries: Optional[List[str]] = None, verbose: bool = False, use_graph: bool = False):
    """Run a set of sample queries or default ones if none provided."""
    if not queries:
        # Default sample queries to test different aspects of the agent
        queries = [
            "Find me a good laptop for programming under $1500",
            "Show me ergonomic chairs with at least 4.5 stars",
            "What's the best gaming PC you have?",
            "I need a standing desk that's adjustable",
            "Show me wireless mice under $50"
        ]
    
    for query in queries:
        if use_graph:
            await test_query_with_graph(query, verbose)
        else:
            await test_query(query, verbose)
        print("\n" + "-" * 80 + "\n")  # Separator between queries

def main():
    """Run the test queries for the ecommerce agent."""
    parser = argparse.ArgumentParser(description="Test the ecommerce agent with sample queries")
    parser.add_argument("--query", "-q", type=str, help="A specific query to test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed product data")
    parser.add_argument("--graph", "-g", action="store_true", help="Test using the full graph workflow instead of direct tool calls")
    args = parser.parse_args()
    
    if args.query:
        # Run a single query if provided
        if args.graph:
            asyncio.run(test_query_with_graph(args.query, args.verbose))
        else:
            asyncio.run(test_query(args.query, args.verbose))
    else:
        # Run all sample queries
        asyncio.run(run_sample_queries(verbose=args.verbose, use_graph=args.graph))
    
    print("✅ Testing complete!")

if __name__ == "__main__":
    main()