"""Configuration management for the MCP Hub project."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class APIConfig:
    """API configuration settings."""
    nebius_api_key: str
    tavily_api_key: str
    nebius_base_url: str = "https://api.studio.nebius.com/v1/"
    current_year: str = "2025"
    
    def __post_init__(self):
        """Validate required API keys."""
        if not self.nebius_api_key:
            raise RuntimeError("NEBIUS_API_KEY is required in your .env file.")
        if not self.tavily_api_key or not self.tavily_api_key.startswith("tvly-"):
            raise RuntimeError("A valid TAVILY_API_KEY is required in your .env file.")

@dataclass
class ModelConfig:
    """Model configuration settings."""
    question_enhancer_model: str = "Qwen/Qwen3-4B-fast"
    llm_processor_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    code_generator_model: str = "Qwen/Qwen2.5-Coder-32B-Instruct-fast"
    orchestrator_model: str = "Qwen/Qwen3-32B-fast"

@dataclass
class AppConfig:
    """Application configuration settings."""
    modal_app_name: str = "my-sandbox-app"
    max_search_results: int = 2
    max_code_generation_attempts: int = 3
    llm_temperature: float = 0.6
    code_gen_temperature: float = 0.2

# Create global configuration instances
api_config = APIConfig(
    nebius_api_key=os.environ.get("NEBIUS_API_KEY", ""),
    tavily_api_key=os.environ.get("TAVILY_API_KEY", ""),
    current_year=os.environ.get("CURRENT_YEAR", "2025")
)

model_config = ModelConfig()
app_config = AppConfig()
