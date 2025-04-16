"""
Script to clear all caches in the ecommerce agent.
Run this script with: poetry run clear-cache
"""

from ecommerce_agent.cache_utils import clear_all_caches

def main():
    """Run the cache clearing utility."""
    print("Clearing ecommerce agent caches...")
    clear_all_caches()

if __name__ == "__main__":
    main()