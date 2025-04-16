"""
Utility module for cache management in the ecommerce agent.
"""

import importlib
import sys
import inspect
from functools import wraps

def clear_all_caches():
    """Clear all caches used in the ecommerce agent."""
    # Clear LRU cache from search_ecommerce function
    try:
        # The original search_ecommerce function is wrapped by the @tool decorator
        # We need to access the wrapped function to clear its cache
        from ecommerce_agent.agent import search_ecommerce
        
        # Try to access the original function through the StructuredTool
        if hasattr(search_ecommerce, 'func'):
            # Some tool implementations store the original function in .func
            original_func = search_ecommerce.func
            if hasattr(original_func, 'cache_clear'):
                original_func.cache_clear()
                print("✓ Cleared search_ecommerce LRU cache via .func attribute")
        elif hasattr(search_ecommerce, '__wrapped__'):
            # Other implementations might use __wrapped__
            wrapped_func = search_ecommerce.__wrapped__
            if hasattr(wrapped_func, 'cache_clear'):
                wrapped_func.cache_clear()
                print("✓ Cleared search_ecommerce LRU cache via __wrapped__ attribute")
        else:
            # As a fallback, we'll try to find the function in the module namespace
            from ecommerce_agent import agent as agent_module
            
            # Look for the original function or any cached functions in the module
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
    
    # Clear imported modules cache if needed
    modules_to_reload = [
        "ecommerce_agent.agent",
        "ecommerce_agent.ecommerce",
        "ecommerce_agent.product_data"
    ]
    
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
            print(f"✓ Reloaded module {module_name}")
    
    # Clear any in-memory data caches
    try:
        from ecommerce_agent.agent import AgentState
        AgentState.last_search_results = []
        print("✓ Cleared agent state search results")
    except ImportError:
        print("× Agent state not available")
    
    print("All caches cleared successfully!")

if __name__ == "__main__":
    clear_all_caches()