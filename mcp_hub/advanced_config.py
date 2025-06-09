"""Advanced configuration management with validation and environment-specific settings."""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from .logging_config import logger

@dataclass
class APIConfig:
    """API configuration with validation."""
    nebius_api_key: str = ""
    nebius_base_url: str = "https://api.studio.nebius.ai/v1/"
    tavily_api_key: str = ""
    
    # API-specific settings
    nebius_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    nebius_max_tokens: int = 1000
    nebius_temperature: float = 0.7
    
    tavily_search_depth: str = "basic"
    tavily_max_results: int = 5
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.nebius_api_key:
            raise ValueError("NEBIUS_API_KEY is required")
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is required")
        
        # Validate numeric ranges
        if not 0.0 <= self.nebius_temperature <= 2.0:
            raise ValueError("nebius_temperature must be between 0.0 and 2.0")
        if self.nebius_max_tokens <= 0:
            raise ValueError("nebius_max_tokens must be positive")
        if self.tavily_max_results <= 0:
            raise ValueError("tavily_max_results must be positive")

@dataclass
class AppConfig:
    """Application configuration."""
    environment: str = "development"  # development, staging, production
    debug: bool = True
    log_level: str = "INFO"
    
    # Gradio settings
    gradio_server_name: str = "0.0.0.0"
    gradio_server_port: int = 7860
    gradio_share: bool = False
    gradio_auth: Optional[tuple] = None
    
    # Performance settings
    max_search_results: int = 10
    max_sub_questions: int = 5
    cache_ttl_seconds: int = 3600
    request_timeout_seconds: int = 30
    
    # Rate limiting
    api_calls_per_second: float = 2.0
    api_burst_size: int = 5
    
    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60
    
    # Monitoring settings
    metrics_retention_hours: int = 24
    health_check_interval_seconds: int = 300  # 5 minutes
    
    def __post_init__(self):
        """Validate application configuration."""
        valid_environments = ["development", "staging", "production"]
        if self.environment not in valid_environments:
            raise ValueError(f"environment must be one of: {valid_environments}")
        
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ValueError(f"log_level must be one of: {valid_log_levels}")
        
        if self.gradio_server_port <= 0 or self.gradio_server_port > 65535:
            raise ValueError("gradio_server_port must be between 1 and 65535")

@dataclass
class SecurityConfig:
    """Security configuration."""
    enable_authentication: bool = False
    allowed_origins: list = field(default_factory=lambda: ["*"])
    api_key_header: str = "X-API-Key"
    rate_limit_per_ip: int = 100  # requests per hour
    max_request_size_mb: int = 10
    
    # Content filtering
    enable_content_filtering: bool = True
    blocked_patterns: list = field(default_factory=list)
    
    def __post_init__(self):
        """Validate security configuration."""
        if self.rate_limit_per_ip <= 0:
            raise ValueError("rate_limit_per_ip must be positive")
        if self.max_request_size_mb <= 0:
            raise ValueError("max_request_size_mb must be positive")

class ConfigManager:
    """Centralized configuration management with environment-specific overrides."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Load environment variables
        self._load_environment_variables()
        
        # Initialize configurations
        self.api_config = self._load_api_config()
        self.app_config = self._load_app_config()
        self.security_config = self._load_security_config()
        
        logger.info(f"Configuration loaded for environment: {self.app_config.environment}")
    
    def _load_environment_variables(self):
        """Load environment variables from .env file if it exists."""
        env_file = Path(".env")
        if env_file.exists():
            from dotenv import load_dotenv
            load_dotenv()
            logger.info("Loaded environment variables from .env file")
    
    def _load_api_config(self) -> APIConfig:
        """Load API configuration from environment and config files."""
        # Start with environment variables
        config_data = {
            "nebius_api_key": os.getenv("NEBIUS_API_KEY", ""),
            "nebius_base_url": os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1/"),
            "tavily_api_key": os.getenv("TAVILY_API_KEY", ""),
            "nebius_model": os.getenv("NEBIUS_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
            "nebius_max_tokens": int(os.getenv("NEBIUS_MAX_TOKENS", "1000")),
            "nebius_temperature": float(os.getenv("NEBIUS_TEMPERATURE", "0.7")),
            "tavily_search_depth": os.getenv("TAVILY_SEARCH_DEPTH", "basic"),
            "tavily_max_results": int(os.getenv("TAVILY_MAX_RESULTS", "5"))
        }
        
        # Override with config file if it exists
        config_file = self.config_dir / "api_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                config_data.update(file_config)
                logger.info("Loaded API configuration from config file")
            except Exception as e:
                logger.warning(f"Failed to load API config file: {e}")
        
        return APIConfig(**config_data)
    
    def _load_app_config(self) -> AppConfig:
        """Load application configuration."""
        environment = os.getenv("ENVIRONMENT", "development")
        
        # Base configuration
        config_data = {
            "environment": environment,
            "debug": environment == "development",
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "gradio_server_name": os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
            "gradio_server_port": int(os.getenv("GRADIO_SERVER_PORT", "7860")),
            "gradio_share": os.getenv("GRADIO_SHARE", "false").lower() == "true",
            "max_search_results": int(os.getenv("MAX_SEARCH_RESULTS", "10")),
            "max_sub_questions": int(os.getenv("MAX_SUB_QUESTIONS", "5")),
            "cache_ttl_seconds": int(os.getenv("CACHE_TTL_SECONDS", "3600")),
            "request_timeout_seconds": int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
        }
        
        # Environment-specific overrides
        env_config_file = self.config_dir / f"app_config_{environment}.json"
        if env_config_file.exists():
            try:
                with open(env_config_file, 'r') as f:
                    env_config = json.load(f)
                config_data.update(env_config)
                logger.info(f"Loaded environment-specific config: {environment}")
            except Exception as e:
                logger.warning(f"Failed to load environment config: {e}")
        
        return AppConfig(**config_data)
    
    def _load_security_config(self) -> SecurityConfig:
        """Load security configuration."""
        config_data = {
            "enable_authentication": os.getenv("ENABLE_AUTH", "false").lower() == "true",
            "rate_limit_per_ip": int(os.getenv("RATE_LIMIT_PER_IP", "100")),
            "max_request_size_mb": int(os.getenv("MAX_REQUEST_SIZE_MB", "10")),
            "enable_content_filtering": os.getenv("ENABLE_CONTENT_FILTERING", "true").lower() == "true"
        }
        
        # Load from config file
        config_file = self.config_dir / "security_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                config_data.update(file_config)
                logger.info("Loaded security configuration from config file")
            except Exception as e:
                logger.warning(f"Failed to load security config: {e}")
        
        return SecurityConfig(**config_data)
    
    def save_config_template(self):
        """Save configuration templates for easy editing."""
        templates = {
            "api_config.json": {
                "nebius_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "nebius_max_tokens": 1000,
                "nebius_temperature": 0.7,
                "tavily_search_depth": "basic",
                "tavily_max_results": 5
            },
            "app_config_development.json": {
                "debug": True,
                "log_level": "DEBUG",
                "gradio_share": False,
                "max_search_results": 5
            },
            "app_config_production.json": {
                "debug": False,
                "log_level": "INFO",
                "gradio_share": False,
                "max_search_results": 10,
                "cache_ttl_seconds": 7200
            },
            "security_config.json": {
                "enable_authentication": False,
                "allowed_origins": ["*"],
                "rate_limit_per_ip": 100,
                "enable_content_filtering": True,
                "blocked_patterns": []
            }
        }
        
        for filename, template in templates.items():
            config_file = self.config_dir / filename
            if not config_file.exists():
                try:
                    with open(config_file, 'w') as f:
                        json.dump(template, f, indent=2)
                    logger.info(f"Created config template: {filename}")
                except Exception as e:
                    logger.error(f"Failed to create config template {filename}: {e}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration (without sensitive data)."""
        return {
            "environment": self.app_config.environment,
            "debug_mode": self.app_config.debug,
            "log_level": self.app_config.log_level,
            "gradio_port": self.app_config.gradio_server_port,
            "cache_ttl": self.app_config.cache_ttl_seconds,
            "max_search_results": self.app_config.max_search_results,
            "authentication_enabled": self.security_config.enable_authentication,
            "content_filtering_enabled": self.security_config.enable_content_filtering,
            "api_endpoints": {
                "nebius": bool(self.api_config.nebius_api_key),
                "tavily": bool(self.api_config.tavily_api_key)
            }
        }
