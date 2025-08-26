"""
Simple Test Configuration for RAGAS Agent Evaluation

Provides only the essential fixtures needed for the simplified RAGAS tests.
"""

import pytest
import os
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path

# Import project modules
from langchain_ollama import ChatOllama
from ragas.llms import LangchainLLMWrapper
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class TestConfiguration:
    """Simple test configuration for Ollama setup"""
    
    # Ollama Configuration (local setup only)
    ollama_llm_model: str = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b-instruct")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Test-specific Model Settings
    test_temperature: float = float(os.getenv("TEST_TEMPERATURE", "0.0"))  # Deterministic for testing
    test_max_tokens: int = int(os.getenv("TEST_MAX_TOKENS", "1000")) if os.getenv("TEST_MAX_TOKENS") else 1000


@pytest.fixture(scope="session")
def test_config() -> TestConfiguration:
    """Test configuration fixture"""
    return TestConfiguration()


@pytest.fixture(scope="session")
def environment_health_check(test_config: TestConfiguration) -> Dict[str, bool]:
    """Check if Ollama server is available"""
    health_status = {}
    
    # Check Ollama availability
    try:
        import requests
        response = requests.get(f"{test_config.ollama_base_url}/api/tags", timeout=5)
        health_status["ollama_server"] = response.status_code == 200
    except Exception:
        health_status["ollama_server"] = False
    
    return health_status


@pytest.fixture(scope="session")
def langchain_llm_ragas_wrapper(test_config: TestConfiguration, environment_health_check: Dict[str, bool]) -> LangchainLLMWrapper:
    """
    Ollama LLM wrapper for RAGAS evaluation
    """
    
    try:
        if not environment_health_check.get("ollama_server", False):
            pytest.skip("Ollama server not available for RAGAS testing")
        
        llm = ChatOllama(
            model=test_config.ollama_llm_model,
            temperature=test_config.test_temperature,
            base_url=test_config.ollama_base_url,
            num_predict=test_config.test_max_tokens
        )
        
        return LangchainLLMWrapper(llm)
        
    except Exception as e:
        pytest.skip(f"Could not initialize RAGAS LLM: {e}")