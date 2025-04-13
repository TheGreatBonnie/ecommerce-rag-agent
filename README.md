# E-Commerce RAG Agent

An intelligent e-commerce assistant powered by Retrieval-Augmented Generation (RAG) that helps users find products and provides detailed recommendations based on natural language queries.

## Overview

This project implements an AI-powered e-commerce agent that:

- Uses vector embeddings to search for relevant products
- Provides intelligent product recommendations based on user queries
- Understands natural language constraints (price limits, ratings, availability)
- Uses LangGraph for agent workflow orchestration
- Integrates with CopilotKit for frontend communication

## Features

- **Vector-based Product Search**: Utilizes sentence transformers to create embeddings for semantic search
- **Intelligent Filtering**: Understands price constraints, ratings, and category preferences from natural language
- **Personalized Recommendations**: Provides tailored product recommendations using LLMs
- **Conversational Interface**: Uses a chat-based interface with LangGraph workflow
- **MongoDB Integration**: Stores product data and vector embeddings for efficient retrieval

## Tech Stack

- **Python 3.10+**
- **LangGraph**: For agent workflow orchestration
- **LangChain**: For LLM interactions and tool usage
- **CopilotKit**: For frontend integration
- **MongoDB**: Vector database for product storage and retrieval
- **Sentence Transformers**: For generating vector embeddings
- **FastAPI**: For serving the agent as an API
- **OpenAI**: For LLM-powered recommendations

## Getting Started

### Prerequisites

- Python 3.10 or higher
- MongoDB Atlas account (for vector search capability)
- OpenAI API key

### Installation

1. Clone the repository:

```bash
git clone https://github.com/TheGreatBonnie/ecommerce-rag-agent.git
cd ecommerce-rag-agent
```

2. Install dependencies using Poetry:

```bash
poetry install
```

Or using pip:

```bash
pip install -e .
```

3. Set up environment variables:

Create a `.env` file in the root directory with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key
MONGODB_USERNAME=your_mongodb_username
MONGODB_PASSWORD=your_mongodb_password
MONGODB_CLUSTER=your_cluster_address
MONGODB_OPTIONS=retryWrites=true&w=majority&appName=Cluster0
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1
```

### Running the Application

Start the FastAPI server:

```bash
poetry run demo
```

Or:

```bash
python -m ecommerce_agent.demo
```

The server will start on http://0.0.0.0:8000 by default.

## Project Structure

- `ecommerce_agent/`
  - `agent.py`: Defines the LangGraph workflow, state, tools, and agent logic
  - `ecommerce.py`: Core e-commerce functionality including MongoDB setup, product search, and recommendations
  - `product_data.py`: Sample product data for demonstration
  - `demo.py`: FastAPI server setup and CopilotKit integration

## Usage Examples

Ask the agent questions like:

- "Find me a good laptop for programming under $1500"
- "What's the best ergonomic chair available?"
- "I need a gaming monitor with at least 4.5 stars"
- "Show me MacBooks with good battery life"

## Development

### Adding New Products

To add new products, edit the `initial_products` list in `product_data.py`. Each product should follow the structure:

```python
{
  "id": "unique_id",
  "name": "Product Name",
  "description": "Product Description",
  "price": 999.99,
  "image": "image_url",
  "category": "Category",
  "rating": 4.5,
  "inStock": True
}
```

### Customizing the Agent

To modify the agent's behavior, edit the system message in `agent.py`. You can also add new tools to enhance the agent's capabilities.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [CopilotKit](https://github.com/copilotkit/copilotkit)
- Uses [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) for embeddings
