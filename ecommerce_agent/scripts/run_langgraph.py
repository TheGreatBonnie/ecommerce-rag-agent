"""
Script to run the ecommerce agent with LangGraph.
This provides a simplified way to run the agent with LangGraph's development server.
"""

import os
import sys

# Set environment variables before imports
os.environ["LANGGRAPH_DEV"] = "true"

# Now import and run LangGraph's CLI
from langgraph_api.cli import app as langgraph_cli

def main():
    """Run the LangGraph development server with the correct environment setup."""
    print("Starting ecommerce agent with LangGraph development server...")
    print("Environment setup: LANGGRAPH_DEV=true")
    # Pass original command line arguments to langgraph_cli
    sys.argv[0] = "langgraph"  # Pretend we're the langgraph command
    if len(sys.argv) == 1:
        sys.argv.append("dev")  # Add 'dev' if no other arguments
    langgraph_cli()

if __name__ == "__main__":
    main()