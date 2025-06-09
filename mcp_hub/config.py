"""Configuration management for the MCP Hub project."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class APIConfig:
    """API configuration settings."""
    # Provider selection
    llm_provider: str = "nebius"  # Options: "nebius", "openai", "anthropic", "huggingface"
    
    # Provider API keys
    nebius_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    huggingface_api_key: str = ""
    
    # Other APIs
    tavily_api_key: str = ""
    
    # Provider URLs
    nebius_base_url: str = "https://api.studio.nebius.com/v1/"
    huggingface_base_url: str = "https://api-inference.huggingface.co"
    
    # Other settings
    current_year: str = "2025"
    
    def __post_init__(self):
        """Validate required API keys based on selected provider."""
        # Always require Tavily for search functionality
        if not self.tavily_api_key or not self.tavily_api_key.startswith("tvly-"):
            raise RuntimeError("A valid TAVILY_API_KEY is required in your .env file.")
        
        # Validate LLM provider selection
        valid_providers = ["nebius", "openai", "anthropic", "huggingface"]
        if self.llm_provider not in valid_providers:
            raise RuntimeError(f"LLM_PROVIDER must be one of: {', '.join(valid_providers)}")
        
        # Validate required API key for selected provider
        if self.llm_provider == "nebius" and not self.nebius_api_key:
            raise RuntimeError("NEBIUS_API_KEY is required when using nebius provider.")
        elif self.llm_provider == "openai" and not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when using openai provider.")
        elif self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required when using anthropic provider.")
        elif self.llm_provider == "huggingface" and not self.huggingface_api_key:
            raise RuntimeError("HUGGINGFACE_API_KEY is required when using huggingface provider.")

@dataclass
class ModelConfig:
    """Model configuration settings."""
    # Default models (Nebius/HuggingFace compatible)
    question_enhancer_model: str = "Qwen/Qwen3-4B-fast"
    llm_processor_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    code_generator_model: str = "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
    orchestrator_model: str = "Qwen/Qwen3-32B-fast"
    
    def get_model_for_provider(self, task: str, provider: str) -> str:
        """Get appropriate model for the given task and provider."""
        
        # Model mappings by provider
        provider_models = {
            "nebius": {
                "question_enhancer": self.question_enhancer_model,
                "llm_processor": self.llm_processor_model,
                "code_generator": self.code_generator_model,
                "orchestrator": self.orchestrator_model,
            },
            "openai": {
                "question_enhancer": "gpt-4.1-nano",
                "llm_processor": "gpt-4.1-nano",
                "code_generator": "gpt-4.1",
                "orchestrator": "gpt-4.1",
            },
            "anthropic": {
                "question_enhancer": "claude-3-5-haiku-latest",#
                "llm_processor": "claude-3-5-sonnet-latest",
                "code_generator": "claude-sonnet-4-0",
                "orchestrator": "claude-sonnet-4-0",
            },
            "huggingface": {
                "question_enhancer": "microsoft/phi-4",
                "llm_processor": "microsoft/phi-4", 
                "code_generator": "Qwen/Qwen2.5-Coder-32B-Instruct",
                "orchestrator": "microsoft/phi-4",
            }
        }
        
        if provider not in provider_models:
            # Fall back to default models
            return getattr(self, f"{task}_model", self.llm_processor_model)
        
        return provider_models[provider].get(task, provider_models[provider]["llm_processor"])

@dataclass
class AppConfig:
    """Application configuration settings."""
    modal_app_name: str = "my-sandbox-app"
    max_search_results: int = 2
    max_code_generation_attempts: int = 3
    llm_temperature: float = 0.6
    code_gen_temperature: float = 0.1

# Create global configuration instances
api_config = APIConfig(
    llm_provider=os.environ.get("LLM_PROVIDER", "nebius"),
    nebius_api_key=os.environ.get("NEBIUS_API_KEY", ""),
    openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    huggingface_api_key=os.environ.get("HUGGINGFACE_API_KEY", ""),
    tavily_api_key=os.environ.get("TAVILY_API_KEY", ""),
    current_year=os.environ.get("CURRENT_YEAR", "2025")
)

model_config = ModelConfig()
app_config = AppConfig()
