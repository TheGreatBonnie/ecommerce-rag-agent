#!/bin/bash
# This script runs before the app starts on Render.com
# It checks for and initializes the MongoDB search index

# Log environment details
echo "Node version: $(node --version)"
echo "Python version: $(python --version)"
echo "Current directory: $(pwd)"

# Set higher OpenAI request timeout for embedding generation
export OPENAI_TIMEOUT_MS=60000
export OPENAI_MAX_RETRIES=3

# Set MongoDB configuration
export MONGODB_CONNECT_TIMEOUT_MS=30000
export MONGODB_SOCKET_TIMEOUT_MS=30000
export MONGODB_SERVER_SELECTION_TIMEOUT_MS=30000

# Check MongoDB connectivity
echo "Testing MongoDB connection..."
python -c "
from ecommerce_agent.ecommerce import setup_mongodb
collection = setup_mongodb()
print(f'MongoDB connection successful. Collection: {collection.name}')
count = collection.count_documents({})
print(f'Product count: {count}')
"

# Log successful initialization
echo "Initialization complete"