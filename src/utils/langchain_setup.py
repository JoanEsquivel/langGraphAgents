"""
Utility functions for the LangGraph chatbot project.
"""
import os
from pathlib import Path
from typing import Tuple, Optional


def load_environment_variables() -> None:
    """
    Load environment variables from .env file if it exists.
    
    This function tries to load environment variables from a .env file
    located in the project root directory. It provides informative
    messages about the loading status.
    """
    try:
        from dotenv import load_dotenv
        
        # Look for .env file in the project root (two levels up from src/utils/)
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✅ Loaded environment variables from {env_path}")
        else:
            print(f"ℹ️  No .env file found at {env_path}")
            print("   LangSmith tracing will use system environment variables if available")
    except ImportError:
        print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
        print("   LangSmith tracing will use system environment variables if available")


def get_langsmith_config() -> Tuple[bool, Optional[str], str]:
    """
    Get LangSmith configuration from environment variables.
    
    Returns:
        Tuple containing:
        - bool: Whether LangSmith tracing is enabled
        - Optional[str]: LangSmith API key (None if not set)
        - str: LangSmith project name (defaults to "default")
    """
    # Check both LANGSMITH_ and LANGCHAIN_ prefixes for compatibility
    langsmith_enabled = (
        os.getenv("LANGSMITH_TRACING", "false").lower() == "true" or
        os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    )
    langsmith_api_key = (
        os.getenv("LANGSMITH_API_KEY") or
        os.getenv("LANGCHAIN_API_KEY")
    )
    langsmith_project = (
        os.getenv("LANGSMITH_PROJECT") or
        os.getenv("LANGCHAIN_PROJECT") or
        "default"
    )
    
    return langsmith_enabled, langsmith_api_key, langsmith_project


def display_langsmith_status() -> None:
    """
    Display the current LangSmith tracing configuration status.
    
    This function checks the environment variables and displays
    informative messages about the LangSmith tracing status.
    """
    langsmith_enabled, langsmith_api_key, langsmith_project = get_langsmith_config()
    
    if langsmith_enabled and langsmith_api_key:
        print(f"🔍 LangSmith tracing enabled for project: {langsmith_project}")
    elif langsmith_enabled:
        print("⚠️  LangSmith tracing enabled but LANGCHAIN_API_KEY not found")
    else:
        print("ℹ️  LangSmith tracing disabled")


def configure_langchain_tracing() -> None:
    """
    Configure LangChain tracing environment variables based on LangSmith config.
    
    This function ensures that LangChain tracing works with LangSmith variables
    by setting the appropriate LANGCHAIN_ environment variables.
    """
    langsmith_enabled, langsmith_api_key, langsmith_project = get_langsmith_config()
    
    if langsmith_enabled:
        # Set LangChain environment variables for tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        if langsmith_api_key:
            os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
        if langsmith_project and langsmith_project != "default":
            os.environ["LANGCHAIN_PROJECT"] = langsmith_project
        
        # Set LangSmith endpoint if provided
        langsmith_endpoint = os.getenv("LANGSMITH_ENDPOINT")
        if langsmith_endpoint:
            os.environ["LANGCHAIN_ENDPOINT"] = langsmith_endpoint


def setup_langsmith() -> None:
    """
    Complete LangSmith setup: load environment variables and display status.
    
    This is a convenience function that handles the complete LangSmith
    setup process including loading environment variables, configuring
    LangChain tracing, and displaying the configuration status.
    """
    load_environment_variables()
    configure_langchain_tracing()
    display_langsmith_status()
