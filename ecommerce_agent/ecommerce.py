"""
This module contains the core e-commerce functionality for the RAG agent.
It handles MongoDB connection, vector embeddings, product search,
and AI-powered product recommendations.
"""

from pymongo import MongoClient 
from pymongo.operations import SearchIndexModel 
import json
from typing import List, Dict
import os
from urllib.parse import quote_plus
from openai import OpenAI
from dotenv import load_dotenv
from ecommerce_agent.product_data import initial_products

# Step 1: Load environment variables from .env file for configuration
# This allows us to keep sensitive information like API keys out of the code
load_dotenv()

# Step 2: Initialize OpenAI client with API key from environment variables
# The client will be used for generating embeddings and AI recommendations
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Step 3: Set embedding model from environment variable or use default
# This model converts text into numerical vectors for similarity comparison
embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

def get_embedding(text: str) -> List[float]:
    """
    Generate vector embeddings for the given text using OpenAI's API.
    Vector embeddings represent text as numerical arrays that capture semantic meaning.
    
    Args:
        text: The text to generate embeddings for
        
    Returns:
        A list of floats representing the embedding vector
    """
    # Step 1: Send text to OpenAI's embedding API
    response = client.embeddings.create(
        input=text,
        model=embedding_model
    )
    # Step 2: Extract the embedding vector from the API response
    return response.data[0].embedding

def prepare_product_for_embedding(product: Dict) -> Dict:
    """
    Prepare a product for embedding by combining relevant fields.
    This creates a searchable representation of the product for vector similarity search.
    
    Args:
        product: Product dictionary with standard fields
        
    Returns:
        Product dictionary with added embedding vector
    """
    # Step 1: Combine name, description, and category for better semantic search
    # This creates a comprehensive text representation of the product
    text_for_embedding = f"{product['name']} {product['description']} {product['category']}"
    
    # Step 2: Generate embedding for the combined text and add to product dict
    return {
        **product,
        "embedding": get_embedding(text_for_embedding)
    }

def setup_mongodb():
    """
    Setup MongoDB connection and create vector search index with improved SSL handling.
    This function establishes a connection to MongoDB Atlas and prepares the database
    for vector-based product searches.
    
    Returns:
        MongoDB collection for products
    """
    # Step 1: Get MongoDB connection parameters from environment variables
    # URL-encode the credentials to handle special characters safely
    username = quote_plus(os.getenv("MONGODB_USERNAME", ""))
    password = quote_plus(os.getenv("MONGODB_PASSWORD", ""))
    cluster = os.getenv("MONGODB_CLUSTER", "cluster0.qeejxg3.mongodb.net")
    options = os.getenv("MONGODB_OPTIONS", "retryWrites=true&w=majority&appName=Cluster0")
    
    # Step 2: Add SSL/TLS options to the connection string if not already present
    # This ensures secure communication with MongoDB Atlas
    ssl_options = "&tls=true"
    if "&tls=" not in options and "&ssl=" not in options:
        options += ssl_options
    
    # Step 3: Construct a properly encoded connection URI
    mongodb_uri = f"mongodb+srv://{username}:{password}@{cluster}/?{options}"
    
    print(f"Connecting to MongoDB...")
    
    # Step 4: Create MongoClient with correct parameter names for newer PyMongo versions
    # Use a two-tier approach: try standard connection first, then fallback if needed
    try:
        # First attempt with standard TLS configuration
        client = MongoClient(
            mongodb_uri,
            tls=True,  # Use tls instead of ssl
            connectTimeoutMS=30000,
            socketTimeoutMS=30000,
            serverSelectionTimeoutMS=30000
        )
        
        # Step 5: Force a connection to verify it works
        client.admin.command('ping')
        print("Successfully connected to MongoDB Atlas")
    except Exception as e:
        print(f"MongoDB connection test failed: {e}")
        # Step 6: Fall back to a more lenient TLS configuration if needed
        try:
            client = MongoClient(
                mongodb_uri,
                tls=True,
                tlsAllowInvalidCertificates=True,  # More lenient TLS validation
                connectTimeoutMS=30000,
                socketTimeoutMS=30000,
                serverSelectionTimeoutMS=30000
            )
            client.admin.command('ping')
            print("Connected to MongoDB with fallback TLS configuration")
        except Exception as e2:
            print(f"Fallback connection also failed: {e2}")
            raise
    
    # Step 7: Get the database and collection references
    db = client["ecommerce_db"]
    collection = db["products"]

    # Step 8: Create vector search index with all product fields
    # This enables semantic search using vector embeddings for product discovery
    index_model = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": True,  # Allow dynamic field mapping
                "fields": {
                    # Define embedding field for vector similarity search
                    "embedding": {
                        "type": "knnVector",
                        "dimensions": 1536,  # Dimensions for OpenAI embeddings
                        "similarity": "cosine"  # Use cosine similarity for comparing vectors
                    },
                    # Define other fields for filtering and sorting operations
                    "id": {
                        "type": "string"
                    },
                    "name": {
                        "type": "string"
                    },
                    "description": {
                        "type": "string"
                    },
                    "price": {
                        "type": "number"
                    },
                    "image": {
                        "type": "string"
                    },
                    "category": {
                        "type": "string"
                    },
                    "rating": {
                        "type": "number"
                    },
                    "inStock": {
                        "type": "boolean"
                    }
                }
            }
        },
        name="vector_index"
    )
    
    # Step 9: Create the search index, ignoring errors if it already exists
    try:
        collection.create_search_index(model=index_model)
    except Exception as e:
        print(f"Error creating index: {e}")

    # Step 10: Return the collection object for use in other functions
    return collection

def search_products(collection, query: str, limit: int = 5):
    """
    Search products using vector similarity with enhanced filtering options.
    This function performs semantic search by converting the query to embeddings
    and finding products with similar vector representations.
    
    Args:
        collection: MongoDB collection containing products
        query: User's search query string
        limit: Maximum number of products to return
        
    Returns:
        List of product dictionaries matching the search criteria
    """
    try:
        # Step 1: Generate embedding for the search query
        print(f"Generating embedding for query: {query}")
        query_embedding = get_embedding(query)
        print(f"Successfully generated embedding for query")
        
        # Step 2: Parse constraints from query using simple text analysis
        # Initialize filter variables to their default values
        price_limit = None
        in_stock_only = False
        min_rating = None
        category_filter = None
        
        # Step 3: Simple parsing for price constraint (e.g., "under $1000")
        if "below" in query.lower() or "under" in query.lower() and "$" in query:
            try:
                price_text = query[query.find("$")+1:]
                price_limit = float(price_text.split()[0].replace(",", ""))
                print(f"Detected price limit: ${price_limit}")
            except Exception as e:
                print(f"Error parsing price: {e}")
                
        # Step 4: Check if query mentions in-stock items
        if "in stock" in query.lower() or "available" in query.lower():
            in_stock_only = True
            
        # Step 5: Check if query mentions rating (e.g., "4 star")
        rating_terms = ["star", "rating", "rated"]
        if any(term in query.lower() for term in rating_terms):
            for i in range(1, 6):  # 1 to 5 stars
                if str(i) in query:
                    min_rating = float(i)
                    print(f"Detected minimum rating: {min_rating}")
                    break
        
        # Step 6: Check for category mentions using a simplified approach
        common_categories = ["laptop", "macbook", "gaming pc", "monitor", "keyboard", 
                            "mouse", "chair", "desk", "accessory", "accessories"]
        for category in common_categories:
            if category in query.lower():
                category_filter = category.capitalize()
                # Special case for multi-word categories
                if category == "gaming pc":
                    category_filter = "Gaming PC"
                print(f"Detected category filter: {category_filter}")
                break

        # Step 7: Create the MongoDB aggregation pipeline
        # This will execute vector search followed by filtering operations
        pipeline = []
        
        # Step 8: First stage - Vector search or find all if vector search fails
        try:
            # Vector search stage using knnBeta for semantic search
            search_stage = {
                "$search": {
                    "index": "vector_index",
                    "knnBeta": {
                        "vector": query_embedding,
                        "path": "embedding",
                        "k": limit * 2  # Get more results than needed to allow for filtering
                    }
                }
            }
            pipeline.append(search_stage)
            print("Added vector search stage to pipeline")
            
            # Step 9: Add a project stage to exclude MongoDB _id field or convert it to string
            pipeline.append({
                "$project": {
                    "_id": 0,  # Exclude the _id field altogether
                    "id": 1,
                    "name": 1,
                    "description": 1,
                    "price": 1,
                    "category": 1,
                    "rating": 1,
                    "inStock": 1,
                    "image": 1,
                    "embedding": 1
                }
            })
            
        except Exception as e:
            print(f"Error creating vector search stage: {e}")
            # If vector search fails, use a simple find operation to return at least some results
            print("Falling back to basic find operation")
            # We'll handle this later if the aggregation fails

        try:
            # Step 10: Execute the pipeline and get results after vector search
            print("Executing MongoDB aggregation pipeline")
            vector_results = list(collection.aggregate(pipeline))
            print(f"Found {len(vector_results)} results from initial vector search")
            
            # Step 11: If vector search failed, fall back to basic find
            if len(vector_results) == 0:
                print("Vector search returned no results, falling back to basic find")
                # Try simple find with no vector search - explicitly exclude _id
                all_results = list(collection.find({}, {'_id': 0, 'embedding': 0}).limit(limit * 2))
                if all_results:
                    print(f"Basic find returned {len(all_results)} results")
                    vector_results = all_results
            
            # Step 12: Apply additional filters based on query constraints
            filter_stages = []
            filtered_results = vector_results
            
            # Step 13: Apply price filter if detected in query
            if price_limit is not None:
                filtered_results = [p for p in filtered_results if p.get('price', float('inf')) <= price_limit]
                print(f"After price filter: {len(filtered_results)} results")
            
            # Step 14: Apply in-stock filter if requested
            if in_stock_only:
                filtered_results = [p for p in filtered_results if p.get('inStock', False)]
                print(f"After in-stock filter: {len(filtered_results)} results")
            
            # Step 15: Apply rating filter if detected
            if min_rating is not None:
                filtered_results = [p for p in filtered_results if p.get('rating', 0) >= min_rating]
                print(f"After rating filter: {len(filtered_results)} results")
            
            # Step 16: Apply category filter if detected
            if category_filter is not None:
                import re
                pattern = re.compile(category_filter, re.IGNORECASE)
                filtered_results = [p for p in filtered_results if 'category' in p and pattern.search(p['category'])]
                print(f"After category filter: {len(filtered_results)} results")
            
            # Step 17: Limit results to requested amount
            final_results = filtered_results[:limit]
            print(f"Final results count: {len(final_results)}")
            
            # Step 18: Ensure all ObjectIds are converted to strings for serialization
            # This prevents JSON serialization errors when returning results
            for product in final_results:
                if '_id' in product:
                    # Convert ObjectId to string if present
                    try:
                        from bson import ObjectId
                        if isinstance(product['_id'], ObjectId):
                            product['_id'] = str(product['_id'])
                            # If 'id' field doesn't exist, use the string version of _id
                            if 'id' not in product:
                                product['id'] = product['_id']
                    except ImportError:
                        # Handle case where bson is not available
                        pass
            
            return final_results
            
        except Exception as e:
            # Step 19: Handle errors during aggregation with fallback strategies
            print(f"Error during MongoDB aggregation: {e}")
            import traceback
            print(traceback.format_exc())
            
            # Multi-layer fallback strategy if primary search fails
            try:
                print("Trying basic find as fallback")
                basic_results = list(collection.find({}, {'_id': 0, 'embedding': 0}).limit(limit))
                print(f"Basic find returned {len(basic_results)} results")
                return basic_results
            except Exception as e2:
                print(f"Basic find also failed: {e2}")
                # Step 20: Final fallback - use products from the initial_products list
                from ecommerce_agent.product_data import initial_products
                print("Using initial_products list as final fallback")
                # Return a subset of products that match the query terms if possible
                query_terms = query.lower().split()
                matches = []
                for product in initial_products:
                    product_text = f"{product['name']} {product['description']} {product['category']}".lower()
                    if any(term in product_text for term in query_terms):
                        matches.append(product)
                
                # If no matches found, return any products up to the limit
                if not matches:
                    return initial_products[:limit]
                return matches[:limit]
            
    except Exception as e:
        # Step 21: Catch-all error handler for the entire function
        print(f"Error in search_products: {e}")
        import traceback
        print(traceback.format_exc())
        return []

def get_product_recommendation(query: str, context_products: List[Dict]) -> str:
    """
    Get AI-generated product recommendations based on the query and retrieved products.
    This function uses OpenAI's GPT model to generate natural language recommendations
    that help users understand which products best meet their needs.
    
    Args:
        query: User's original search query
        context_products: List of product dictionaries to recommend from
        
    Returns:
        Formatted text recommendation with product comparisons and links
    """
    # Step 1: Handle the case where no products were found
    if not context_products:
        return "I apologize, but I couldn't find any products matching your criteria."
    
    try:
        # Step 2: Track timing for performance monitoring
        import time
        start_time = time.time()
        
        # Step 3: Create context with detailed product information for the LLM
        context = "\n\nAVAILABLE PRODUCTS:\n"
        for i, p in enumerate(context_products, 1):
            product_link = f"/products/{p['id']}"
            context += f"""
Product {i}:
- Name: {p['name']}
- Description: {p['description']}
- Price: ${p['price']}
- Category: {p['category']}
- Rating: {p['rating']} out of 5
- Link: {product_link}
"""
        
        # Step 4: Comprehensive prompt to guide the LLM in generating useful recommendations
        prompt = f"""You are a knowledgeable e-commerce assistant. Your task is to recommend products from the AVAILABLE PRODUCTS list below based on the customer's query.
ONLY recommend products that are listed in the AVAILABLE PRODUCTS section. DO NOT make up or suggest products that are not in this list.

{context}

Customer Query: {query}

Please provide a detailed recommendation that:
1. ONLY discusses the products listed above
2. Compares relevant features and prices
3. Explains why specific products would or wouldn't meet the customer's needs
4. Considers any price constraints mentioned in the query
5. IMPORTANT: For each product you recommend, include a clickable link in the format: [Product Name](/products/id) - where 'id' is the product ID

Response Format:
1. Start with a brief introduction
2. List and compare the most relevant products from the available options, including links to each product in markdown format
3. Conclude with a specific recommendation
4. ALWAYS format product links as [Product Name](/products/id)"""

        # Step 5: Make API call to OpenAI for recommendation generation
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a knowledgeable e-commerce assistant. Include markdown links to products in your recommendations. Format as [Product Name](/products/id)."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800  # Allow for detailed responses
        )
        
        # Step 6: Log timing for performance monitoring
        elapsed_time = time.time() - start_time
        print(f"Recommendation generation time: {elapsed_time:.2f}s")
        
        # Step 7: Extract the recommendation text from the API response
        recommendation = response.choices[0].message.content
        
        # Step 8: Check if response is too large and truncate if needed
        if len(recommendation) > 10000:
            print(f"Warning: Large recommendation generated ({len(recommendation)} chars)")
            recommendation = recommendation[:9000] + "\n\n[Note: This recommendation was truncated for better performance]"
            
        return recommendation
    except Exception as e:
        # Step 9: Handle errors in recommendation generation
        print(f"Error in product recommendation: {str(e)}")
        # Return a useful error message to the user
        return f"I apologize, but I encountered an error while generating product recommendations. Please try again with a different query. Error details: {str(e)}"

def main():
    """
    Main entry point for the e-commerce system.
    Sets up MongoDB connection and populates it with products if empty.
    This function initializes the entire system and prepares it for product searches.
    
    Returns:
        MongoDB collection object for use by other functions
    """
    # Step 1: Setup MongoDB connection and create necessary indexes
    collection = setup_mongodb()
    
    # Step 2: Check if products already exist in the database
    existing_count = collection.count_documents({})
    
    # Step 3: Only populate the database if it's empty - avoids duplicate inserts
    if existing_count == 0:
        print(f"Database empty. Inserting {len(initial_products)} products...")
        
        # Step 4: Use multiprocessing for parallel embedding generation to improve performance
        import concurrent.futures
        
        # Define a function to prepare products in parallel
        def prepare_product_batch(products_batch):
            """Helper function to process a batch of products with embeddings"""
            return [prepare_product_for_embedding(p) for p in products_batch]
        
        # Step 5: Split products into batches for parallel processing
        batch_size = 10  # Adjust based on your API rate limits
        product_batches = [initial_products[i:i+batch_size] 
                          for i in range(0, len(initial_products), batch_size)]
        
        products_with_embeddings = []
        
        # Step 6: Process batches in parallel using ThreadPoolExecutor
        # This speeds up the embedding generation process significantly
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            batch_results = executor.map(prepare_product_batch, product_batches)
            
            # Step 7: Flatten results from all batches into a single list
            for batch_result in batch_results:
                products_with_embeddings.extend(batch_result)
        
        # Step 8: Insert all products in one operation for efficiency
        result = collection.insert_many(products_with_embeddings)
        print(f"Successfully inserted {len(result.inserted_ids)} products")
    else:
        print(f"Database already contains {existing_count} products. Skipping insertion.")
    
    # Step 9: Return the collection for other functions to use
    return collection

if __name__ == "__main__":
    main()

"""
=== E-COMMERCE RAG AGENT SYSTEM SUMMARY ===

This file implements a complete Retrieval-Augmented Generation (RAG) system for e-commerce 
product search and recommendations. Here's how the system works:

SYSTEM ARCHITECTURE:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   User Query    │ -> │  Vector Search   │ -> │  AI Recommendation  │
│   "gaming laptop"│    │  (MongoDB Atlas) │    │    (OpenAI GPT)     │
└─────────────────┘    └──────────────────┘    └─────────────────────┘

KEY COMPONENTS:

1. VECTOR EMBEDDINGS SYSTEM
   - Converts product descriptions into numerical vectors using OpenAI embeddings
   - Enables semantic search (finds "gaming laptop" even if query is "portable computer for games")
   - Uses text-embedding-3-small model with 1536 dimensions

2. MONGODB ATLAS VECTOR DATABASE
   - Stores products with their vector embeddings
   - Uses MongoDB Atlas Vector Search for k-nearest neighbor (KNN) similarity search
   - Includes fallback mechanisms for reliability

3. INTELLIGENT QUERY PARSING
   - Extracts price constraints ("under $1000")
   - Detects stock requirements ("in stock items")
   - Identifies rating filters ("4+ star products")
   - Recognizes category preferences ("laptops", "gaming PC")

4. AI-POWERED RECOMMENDATIONS
   - Uses GPT-4-turbo to generate natural language product recommendations
   - Compares products based on features, price, and user needs
   - Includes clickable product links in markdown format

5. ROBUST ERROR HANDLING
   - Multiple fallback strategies if vector search fails
   - Graceful degradation to basic database queries
   - Final fallback to hardcoded product list
   - Performance monitoring and logging

WORKFLOW:
1. User submits search query
2. System generates vector embedding for the query
3. Performs semantic similarity search in MongoDB
4. Applies additional filters (price, rating, category, stock)
5. Sends filtered results to GPT-4 for natural language recommendations
6. Returns formatted recommendations with product links

PERFORMANCE OPTIMIZATIONS:
- Parallel embedding generation using ThreadPoolExecutor
- Batch processing for database operations
- Connection pooling and timeout management
- Response size monitoring and truncation

DEPENDENCIES:
- pymongo: MongoDB database operations
- openai: Embeddings and chat completions
- python-dotenv: Environment variable management
- concurrent.futures: Parallel processing

ENVIRONMENT VARIABLES REQUIRED:
- OPENAI_API_KEY: OpenAI API access
- MONGODB_USERNAME: Database authentication
- MONGODB_PASSWORD: Database authentication
- MONGODB_CLUSTER: Database cluster URL
- EMBEDDING_MODEL: (Optional) Embedding model selection

This system demonstrates a production-ready implementation of RAG architecture,
combining vector similarity search with large language model capabilities to
provide intelligent e-commerce product recommendations.
"""