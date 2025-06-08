"""Common test fixtures and configuration."""

import pytest
import asyncio
import os
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, Generator

# Mock environment variables for testing - set them globally before any imports
TEST_ENV_VARS = {
    "TAVILY_API_KEY": "tvly-test-key-12345",
    "NEBIUS_API_KEY": "test-nebius-key",
    "OPENAI_API_KEY": "test-openai-key", 
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "HUGGINGFACE_API_KEY": "test-hf-key",
    "LLM_PROVIDER": "nebius"
}

# Set environment variables immediately
for key, value in TEST_ENV_VARS.items():
    os.environ[key] = value

@pytest.fixture
def mock_tavily_client():
    """Mock Tavily client for web search tests."""
    mock_client = Mock()
    mock_client.search.return_value = {
        "results": [
            {
                "title": "Test Result 1",
                "url": "https://example.com/1",
                "content": "Test content 1",
                "score": 0.9
            },
            {
                "title": "Test Result 2", 
                "url": "https://example.com/2",
                "content": "Test content 2",
                "score": 0.8
            }
        ],
        "answer": "Test search summary"
    }
    return mock_client

@pytest.fixture
def mock_llm_response():
    """Mock LLM completion response."""
    return '{"sub_questions": ["Question 1?", "Question 2?", "Question 3?"]}'

@pytest.fixture
def mock_modal_sandbox():
    """Mock Modal sandbox for code execution tests."""
    mock_sandbox = Mock()
    mock_sandbox.exec.return_value = Mock(stdout="Test output", stderr="", returncode=0)
    return mock_sandbox

@pytest.fixture
def sample_user_request():
    """Sample user request for testing."""
    return "Create a Python script to analyze CSV data and generate charts"

@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        {
            "title": "Python Data Analysis Tutorial",
            "url": "https://example.com/pandas-tutorial",
            "content": "Learn how to analyze CSV data with pandas and matplotlib...",
            "score": 0.95
        },
        {
            "title": "Chart Generation with Python",
            "url": "https://example.com/charts",
            "content": "Create stunning charts and visualizations...",
            "score": 0.87
        }
    ]

@pytest.fixture
def sample_code():
    """Sample Python code for testing."""
    return '''
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('data.csv')

# Generate chart
df.plot(kind='bar')
plt.show()
'''

@pytest.fixture
def mock_config():
    """Mock configuration objects."""
    api_config = Mock()
    api_config.tavily_api_key = "tvly-test-key"
    api_config.llm_provider = "nebius"
    api_config.nebius_api_key = "test-nebius-key"
    
    model_config = Mock()
    model_config.get_model_for_provider.return_value = "meta-llama/llama-3.1-8b-instruct"
    
    return api_config, model_config

@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

class MockAgent:
    """Base mock agent class for testing."""
    def __init__(self, name: str):
        self.name = name
        self.call_count = 0
        
    def __call__(self, *args, **kwargs):
        self.call_count += 1
        return {"success": True, "agent": self.name, "calls": self.call_count}

@pytest.fixture  
def mock_agents():
    """Mock agent instances for orchestrator testing."""
    return {
        "question_enhancer": MockAgent("question_enhancer"),
        "web_search": MockAgent("web_search"),
        "llm_processor": MockAgent("llm_processor"), 
        "citation_formatter": MockAgent("citation_formatter"),
        "code_generator": MockAgent("code_generator"),
        "code_runner": MockAgent("code_runner")
    }

@pytest.fixture
def disable_advanced_features():
    """Disable advanced features for basic testing."""
    with patch('app.ADVANCED_FEATURES_AVAILABLE', False):
        yield