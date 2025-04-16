"""
This serves the "sample_agent" agent. This is an example of self-hosting an agent
through our FastAPI integration. However, you can also host in LangGraph platform.
"""

import os
from dotenv import load_dotenv

# Set environment variable for handling blocking operations
os.environ["BG_JOB_ISOLATED_LOOPS"] = "true"

load_dotenv() # pylint: disable=wrong-import-position

from fastapi import FastAPI
import uvicorn # type: ignore
from copilotkit.integrations.fastapi import add_fastapi_endpoint
from copilotkit import CopilotKitRemoteEndpoint, LangGraphAgent
from ecommerce_agent.agent import graph

app = FastAPI()
sdk = CopilotKitRemoteEndpoint(
    agents=[
        LangGraphAgent(
            name="ecommerce_agent",
            description="An example agent to use as a starting point for your own agent.",
            graph=graph,
        )
    ],
)

add_fastapi_endpoint(app, sdk, "/copilotkit")

# add new route for health check
@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}

def main():
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "ecommerce_agent.demo:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        reload_dirs=(
            ["."] +
            (["../../../../sdk-python/copilotkit"]
             if os.path.exists("../../../../sdk-python/copilotkit")
             else []
             )
        )
    )
