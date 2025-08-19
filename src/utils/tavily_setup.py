"""
Tavily search tool configuration and setup utilities.
"""
import os
from typing import Optional
from langchain_tavily import TavilySearch


def get_tavily_config() -> Optional[str]:
    """
    Get Tavily API key from environment variables.
    
    Returns:
        Optional[str]: Tavily API key (None if not set)
    """
    return os.getenv("TAVILY_API_KEY")


def create_tavily_tool(max_results: int = 3) -> Optional[TavilySearch]:
    """
    Create and configure a Tavily search tool.
    
    Args:
        max_results (int): Maximum number of search results to return
        
    Returns:
        Optional[TavilySearch]: Configured Tavily tool or None if API key not available
    """
    api_key = get_tavily_config()
    
    if not api_key:
        print("⚠️  TAVILY_API_KEY not found - Tavily search will be unavailable")
        return None
    
    try:
        tavily_tool = TavilySearch(
            max_results=max_results,
            api_key=api_key
        )
        print(f"✅ Tavily search tool configured (max_results={max_results})")
        return tavily_tool
    except Exception as e:
        print(f"❌ Error creating Tavily tool: {e}")
        return None


def display_tavily_status() -> None:
    """
    Display the current Tavily configuration status.
    """
    api_key = get_tavily_config()
    
    if api_key:
        # Mask the API key for security
        masked_key = f"{api_key[:8]}..." if len(api_key) > 8 else "***"
        print(f"🔍 Tavily API key configured: {masked_key}")
    else:
        print("ℹ️  Tavily API key not configured")
        print("   Set TAVILY_API_KEY environment variable to enable web search")


def setup_tavily(max_results: int = 3) -> Optional[TavilySearch]:
    """
    Complete Tavily setup: display status and create tool.
    
    Args:
        max_results (int): Maximum number of search results to return
        
    Returns:
        Optional[TavilySearch]: Configured Tavily tool or None if not available
    """
    display_tavily_status()
    tool = create_tavily_tool(max_results)
    
    # Test the tool briefly if available (following tutorial pattern)
    if tool:
        print(f"✅ Tavily tool ready for web search")
    
    return tool
