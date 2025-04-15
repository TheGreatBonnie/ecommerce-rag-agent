from pymongo import MongoClient 
from pymongo.operations import SearchIndexModel 
import json
from typing import List, Dict
import os
from urllib.parse import quote_plus
from openai import OpenAI
from dotenv import load_dotenv
from ecommerce_agent.product_data import initial_products

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set embedding model
embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

def get_embedding(text: str) -> List[float]:
    """Generate vector embeddings for the given text using OpenAI's API."""
    response = client.embeddings.create(
        input=text,
        model=embedding_model
    )
    return response.data[0].embedding

def prepare_product_for_embedding(product: Dict) -> Dict:
    """Prepare a product for embedding by combining relevant fields."""
    # Combine name, description, and category for better semantic search
    text_for_embedding = f"{product['name']} {product['description']} {product['category']}"
    return {
        **product,
        "embedding": get_embedding(text_for_embedding)
    }

def setup_mongodb():
    """Setup MongoDB connection and create vector search index with improved SSL handling."""
    # Get MongoDB connection parameters
    username = quote_plus(os.getenv("MONGODB_USERNAME", ""))
    password = quote_plus(os.getenv("MONGODB_PASSWORD", ""))
    cluster = os.getenv("MONGODB_CLUSTER", "cluster0.qeejxg3.mongodb.net")
    options = os.getenv("MONGODB_OPTIONS", "retryWrites=true&w=majority&appName=Cluster0")
    
    # Add SSL/TLS options to the connection string
    ssl_options = "&tls=true"
    if "&tls=" not in options and "&ssl=" not in options:
        options += ssl_options
    
    # Construct a properly encoded connection URI
    mongodb_uri = f"mongodb+srv://{username}:{password}@{cluster}/?{options}"
    
    print(f"Connecting to MongoDB...")
    
    # Create MongoClient with correct parameter names for newer PyMongo versions
    try:
        client = MongoClient(
            mongodb_uri,
            tls=True,  # Use tls instead of ssl
            connectTimeoutMS=30000,
            socketTimeoutMS=30000,
            serverSelectionTimeoutMS=30000
        )
        
        # Force a connection to verify it works
        client.admin.command('ping')
        print("Successfully connected to MongoDB Atlas")
    except Exception as e:
        print(f"MongoDB connection test failed: {e}")
        # Fall back to a more lenient TLS configuration if needed
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
    
    db = client["ecommerce_db"]
    collection = db["products"]

    # Create vector search index with all product fields
    index_model = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": True,
                "fields": {
                    "embedding": {
                        "type": "knnVector",
                        "dimensions": 1536,
                        "similarity": "cosine"
                    },
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
    
    try:
        collection.create_search_index(model=index_model)
    except Exception as e:
        print(f"Error creating index: {e}")

    return collection

def search_products(collection, query: str, limit: int = 5):
    """Search products using vector similarity with enhanced filtering options."""
    # Generate embedding for the query
    query_embedding = get_embedding(query)
    
    # Parse constraints from query
    price_limit = None
    in_stock_only = False
    min_rating = None
    category_filter = None
    
    # Simple parsing for price constraint
    if "below" in query.lower() and "$" in query:
        try:
            price_text = query[query.find("$")+1:]
            price_limit = float(price_text.split()[0].replace(",", ""))
        except:
            pass
            
    # Check if query mentions in-stock items
    if "in stock" in query.lower() or "available" in query.lower():
        in_stock_only = True
        
    # Check if query mentions rating
    rating_terms = ["star", "rating", "rated"]
    if any(term in query.lower() for term in rating_terms):
        for i in range(1, 6):  # 1 to 5 stars
            if str(i) in query:
                min_rating = float(i)
                break
    
    # Check for category mentions - this is simplified and could be improved
    common_categories = ["laptop", "macbook", "gaming pc", "monitor", "keyboard", 
                        "mouse", "chair", "desk", "accessory", "accessories"]
    for category in common_categories:
        if category in query.lower():
            category_filter = category.capitalize()
            # Special case for multi-word categories
            if category == "gaming pc":
                category_filter = "Gaming PC"
            break

    # Create the pipeline
    pipeline = []
    
    # First stage: Vector search
    search_stage = {
        "$search": {
            "index": "vector_index",
            "knnBeta": {
                "vector": query_embedding,
                "path": "embedding",
                "k": limit * 2
            }
        }
    }
    pipeline.append(search_stage)

    try:
        # Get results after vector search
        vector_results = list(collection.aggregate(pipeline))
        
        # Apply additional filters
        filter_stages = []
        
        # Price filter
        if price_limit is not None:
            filter_stages.append({
                "$match": {
                    "price": {"$lte": price_limit}
                }
            })
        
        # In-stock filter
        if in_stock_only:
            filter_stages.append({
                "$match": {
                    "inStock": True
                }
            })
        
        # Rating filter
        if min_rating is not None:
            filter_stages.append({
                "$match": {
                    "rating": {"$gte": min_rating}
                }
            })
            
        # Category filter
        if category_filter is not None:
            filter_stages.append({
                "$match": {
                    "category": {"$regex": category_filter, "$options": "i"}
                }
            })
        
        # Apply all filters to the pipeline
        pipeline.extend(filter_stages)

        # Limit results
        pipeline.append({"$limit": limit})

        # Project fields (exclude _id and embedding)
        pipeline.append({
            "$project": {
                "_id": 0,
                "embedding": 0
            }
        })
        
        final_results = list(collection.aggregate(pipeline))
        return final_results
        
    except Exception as e:
        print(f"Error during product search: {e}")
        return []

def get_product_recommendation(query: str, context_products: List[Dict]) -> str:
    """Get AI-generated product recommendations based on the query and retrieved products."""
    if not context_products:
        return "I apologize, but I couldn't find any products matching your criteria."
    
    context = "\n\nAVAILABLE PRODUCTS:\n"
    for i, p in enumerate(context_products, 1):
        context += f"""
Product {i}:
- Name: {p['name']}
- Description: {p['description']}
- Price: ${p['price']}
- Category: {p['category']}
- Rating: {p['rating']} out of 5
"""
    
    prompt = f"""You are a knowledgeable e-commerce assistant. Your task is to recommend products from the AVAILABLE PRODUCTS list below based on the customer's query.
ONLY recommend products that are listed in the AVAILABLE PRODUCTS section. DO NOT make up or suggest products that are not in this list.

{context}

Customer Query: {query}

Please provide a detailed recommendation that:
1. ONLY discusses the products listed above
2. Compares relevant features and prices
3. Explains why specific products would or wouldn't meet the customer's needs
4. Considers any price constraints mentioned in the query

Response Format:
1. Start with a brief introduction
2. List and compare the most relevant products from the available options
3. Conclude with a specific recommendation"""

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a knowledgeable e-commerce assistant. Only recommend products from the provided list. Never suggest products that aren't in the available products list."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    
    return response.choices[0].message.content # type: ignore

def main():
    collection = setup_mongodb()
    
    # Check if products already exist in the database
    existing_count = collection.count_documents({})
    
    # Only populate the database if it's empty
    if existing_count == 0:
        print(f"Database empty. Inserting {len(initial_products)} products...")
        
        # Use multiprocessing for parallel embedding generation
        import concurrent.futures
        
        # Define a function to prepare products in parallel
        def prepare_product_batch(products_batch):
            return [prepare_product_for_embedding(p) for p in products_batch]
        
        # Split products into batches for parallel processing
        batch_size = 10  # Adjust based on your API rate limits
        product_batches = [initial_products[i:i+batch_size] 
                          for i in range(0, len(initial_products), batch_size)]
        
        products_with_embeddings = []
        
        # Process batches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            batch_results = executor.map(prepare_product_batch, product_batches)
            
            # Flatten results from all batches
            for batch_result in batch_results:
                products_with_embeddings.extend(batch_result)
        
        # Insert all products in one operation
        result = collection.insert_many(products_with_embeddings)
        print(f"Successfully inserted {len(result.inserted_ids)} products")
    else:
        print(f"Database already contains {existing_count} products. Skipping insertion.")
    
    # Return the collection for other functions to use
    return collection

if __name__ == "__main__":
    main()