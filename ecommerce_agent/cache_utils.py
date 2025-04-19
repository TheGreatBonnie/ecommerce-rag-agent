"""
Utility module for cache management in the ecommerce agent.
This module provides functions to clear caches used throughout the application,
which helps maintain up-to-date search results and prevents stale data.
"""

import importlib
import sys
import inspect
from functools import wraps

def clear_all_caches():
    """
    Clear all caches used in the ecommerce agent.
    
    This function:
    1. Clears the LRU cache from the search_ecommerce function
    2. Reloads key modules to refresh any module-level caches
    3. Resets any in-memory data structures that might contain cached data
    
    Returns:
        None, but prints status messages about cleared caches
    """
    # Clear LRU cache from search_ecommerce function
    try:
        # The search_ecommerce function is decorated with @tool and @lru_cache
        # We need to find and clear its underlying cache to ensure fresh results
        from ecommerce_agent.agent import search_ecommerce
        
        # Attempt multiple approaches to find the cache_clear method
        # since tool decorators can wrap functions in different ways
        if hasattr(search_ecommerce, 'func'):
            # Some tool implementations store the original function in .func
            original_func = search_ecommerce.func
            if hasattr(original_func, 'cache_clear'):
                original_func.cache_clear()
                print("✓ Cleared search_ecommerce LRU cache via .func attribute")
        elif hasattr(search_ecommerce, '__wrapped__'):
            # Other implementations might use the standard __wrapped__ attribute
            wrapped_func = search_ecommerce.__wrapped__
            if hasattr(wrapped_func, 'cache_clear'):
                wrapped_func.cache_clear()
                print("✓ Cleared search_ecommerce LRU cache via __wrapped__ attribute")
        else:
            # As a fallback, search the entire module for cached functions
            from ecommerce_agent import agent as agent_module
            
            # Look for any object with a cache_clear method
            cache_cleared = False
            for name, obj in inspect.getmembers(agent_module):
                if hasattr(obj, 'cache_clear') and callable(obj.cache_clear):
                    obj.cache_clear()
                    print(f"✓ Cleared cache for {name}")
                    cache_cleared = True
            
            if not cache_cleared:
                print("× Unable to clear search_ecommerce cache - function wrapper structure not recognized")
    except Exception as e:
        print(f"× Error clearing search_ecommerce cache: {e}")
    
    # Clear imported modules cache by reloading key modules
    # This ensures any module-level variables or caches are refreshed
    modules_to_reload = [
        "ecommerce_agent.agent",
        "ecommerce_agent.ecommerce",
        "ecommerce_agent.product_data"
    ]
    
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
            print(f"✓ Reloaded module {module_name}")
    
    # Clear any in-memory data caches in the AgentState class
    # This ensures the agent uses fresh data for new queries
    try:
        from ecommerce_agent.agent import AgentState
        if hasattr(AgentState, 'last_search_results'):
            AgentState.last_search_results = []
            print("✓ Cleared agent state search results")
        
        # Reset other state variables that might contain cached data
        if hasattr(AgentState, 'search_history'):
            AgentState.search_history = []
            print("✓ Cleared agent search history")
            
        if hasattr(AgentState, 'current_query'):
            AgentState.current_query = ""
            print("✓ Reset agent current query")
    except ImportError:
        print("× Agent state not available")
    
    print("All caches cleared successfully!")

if __name__ == "__main__":
    # This allows the script to be run directly to clear all caches
    clear_all_caches()