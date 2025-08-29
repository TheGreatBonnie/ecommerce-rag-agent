"""
This serves the e-commerce RAG agent through a FastAPI web server.
This is a production-ready example of self-hosting an AI agent using CopilotKit's
FastAPI integration. The server provides REST endpoints for agent communication
and can be deployed to any cloud platform or run locally for development.

Alternative deployment options:
- LangGraph Platform (managed hosting)  
- Docker containerization
- Cloud platforms (AWS, GCP, Azure)
"""

import os
from dotenv import load_dotenv

# Step 1: Set environment variable for handling blocking operations
# This prevents deadlocks when the agent performs I/O operations like database queries
os.environ["BG_JOB_ISOLATED_LOOPS"] = "true"

# Step 2: Load environment variables from .env file
# This must happen early to ensure all configuration is available before imports
load_dotenv() # pylint: disable=wrong-import-position

# Step 3: Import required libraries after environment setup
from fastapi import FastAPI
import uvicorn # type: ignore
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitRemoteEndpoint, LangGraphAgent
from ecommerce_agent.agent import graph

# Step 4: Create FastAPI application instance
# FastAPI provides automatic API documentation, request validation, and async support
app = FastAPI()

# Step 5: Configure CopilotKit SDK with the e-commerce agent
# This creates a remote endpoint that can handle agent requests from frontend clients
sdk = CopilotKitRemoteEndpoint(
    agents=[
        LangGraphAgent(
            name="ecommerce_agent",  # Unique identifier for this agent
            description="An intelligent e-commerce assistant that helps users find products and provides AI-powered recommendations using vector search and natural language processing.",
            graph=graph,  # The LangGraph workflow defined in agent.py
        )
    ],
)

# Step 6: Add CopilotKit endpoint to FastAPI application
# This creates the /copilotkit endpoint that frontend clients will communicate with
add_fastapi_endpoint(app, sdk, "/copilotkit")

# Step 7: Add health check endpoint for monitoring and deployment
@app.get("/health")
def health():
    """
    Health check endpoint for monitoring system status.
    This endpoint is commonly used by:
    - Load balancers to check if the service is healthy
    - Container orchestrators (Docker, Kubernetes) for health probes  
    - Monitoring systems for uptime tracking
    - CI/CD pipelines for deployment verification
    
    Returns:
        dict: Simple status response indicating the service is operational
    """
    return {"status": "ok"}

def main():
    """
    Main entry point for running the FastAPI server with uvicorn.
    This function configures and starts the web server that hosts the e-commerce agent.
    
    The server configuration includes:
    - Dynamic port configuration from environment variables
    - Hot reloading for development
    - Host binding for external access
    - Development-friendly reload directories
    """
    # Step 1: Get port from environment variable with fallback to 8000
    # This allows flexible deployment across different environments
    port = int(os.getenv("PORT", "8000"))
    
    # Step 2: Start uvicorn server with development-optimized settings
    uvicorn.run(
        "ecommerce_agent.demo:app",  # Module and application reference
        host="0.0.0.0",  # Bind to all network interfaces for external access
        port=port,  # Use configured port number
        reload=True,  # Enable hot reloading for development
        reload_dirs=(
            # Step 3: Configure reload directories for file watching
            ["."] +  # Watch current directory for changes
            (["../../../../sdk-python/copilotkit"]  # Watch CopilotKit SDK if available
             if os.path.exists("../../../../sdk-python/copilotkit")
             else []  # Skip if SDK directory doesn't exist
             )
        )
    )

# Step 4: Execute main function when script is run directly
if __name__ == "__main__":
    main()

"""
=== FASTAPI SERVER DEPLOYMENT SUMMARY ===

This file serves as the production deployment interface for the e-commerce RAG agent,
providing a RESTful API through FastAPI that frontend applications can communicate with.

SERVER ARCHITECTURE:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Frontend App   │ -> │   FastAPI Server │ -> │  E-commerce Agent   │
│ (React/Vue/etc) │    │ (REST Endpoints) │    │   (LangGraph)       │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
        │                       │                        │
        v                       v                        v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   HTTP/WebSocket│    │   CopilotKit     │    │   MongoDB Vector    │
│   Communication │    │   Integration    │    │   Search + OpenAI   │
└─────────────────┘    └──────────────────┘    └─────────────────────┘

KEY COMPONENTS:

1. FASTAPI APPLICATION
   - Automatic API documentation at /docs and /redoc
   - Request/response validation with Pydantic models
   - Async support for high-performance I/O operations
   - Built-in OpenAPI specification generation

2. COPILOTKIT INTEGRATION
   - Remote endpoint configuration for agent communication
   - WebSocket support for real-time conversation streaming
   - State synchronization between frontend and backend
   - Agent lifecycle management and error handling

3. LANGGRAPH AGENT HOSTING
   - Self-hosted agent deployment (alternative to LangGraph Platform)
   - Custom agent configuration with name and description
   - Graph-based workflow execution with state persistence
   - Tool integration for product search and recommendations

4. DEVELOPMENT FEATURES
   - Hot reloading for rapid development cycles
   - Environment variable configuration for flexibility
   - Health check endpoint for monitoring and deployment
   - SDK directory watching for CopilotKit development

5. PRODUCTION READINESS
   - Configurable port binding from environment variables
   - Host binding for external network access (0.0.0.0)
   - Health monitoring endpoint for load balancers
   - Isolated event loops for blocking operation handling

DEPLOYMENT OPTIONS:

Local Development:
- Run with: python -m ecommerce_agent.demo
- Access at: http://localhost:8000
- API docs at: http://localhost:8000/docs

Docker Deployment:
- Build container with FastAPI and dependencies
- Expose port 8000 or use PORT environment variable
- Mount .env file for configuration

Cloud Deployment:
- AWS: Elastic Beanstalk, ECS, or Lambda
- GCP: Cloud Run, App Engine, or Compute Engine  
- Azure: App Service, Container Instances, or Functions
- Heroku: Direct deployment with Procfile

Load Balancer Configuration:
- Health check: GET /health
- Expected response: {"status": "ok"}
- Timeout: 30 seconds recommended

ENVIRONMENT VARIABLES:
- PORT: Server port (default: 8000)
- OPENAI_API_KEY: Required for AI functionality
- MONGODB_USERNAME: Database authentication
- MONGODB_PASSWORD: Database authentication
- MONGODB_CLUSTER: Database cluster URL
- BG_JOB_ISOLATED_LOOPS: Set to "true" for async safety

API ENDPOINTS:
- POST /copilotkit: Main agent communication endpoint
- GET /health: Health check for monitoring
- GET /docs: Interactive API documentation
- GET /redoc: Alternative API documentation

This server provides a complete, production-ready hosting solution for the
e-commerce RAG agent, with built-in monitoring, documentation, and deployment
flexibility across various cloud platforms and container orchestrators.
"""
