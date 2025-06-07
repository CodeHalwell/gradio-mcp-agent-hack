# MCP Hub Project - Multi-Agent AI Research & Code Assistant

## Overview

The MCP (Model Context Protocol) Hub is an advanced multi-agent research and code assistant built using Gradio's MCP server functionality. This project demonstrates how to build a sophisticated workflow of interconnected AI agents that collaborate to provide comprehensive research capabilities and generate executable Python code.

The system orchestrates a complete AI-powered research and development workflow:

1. **Question Enhancement**: Intelligently breaks down complex queries into focused sub-questions
2. **Web Search**: Performs targeted web searches using Tavily API for up-to-date information
3. **LLM Processing**: Processes and summarizes information using configurable LLM providers
4. **Citation Management**: Automatically extracts and formats APA-style citations
5. **Context Integration**: Combines research findings into comprehensive, grounded context
6. **Code Generation**: Creates Python code based on research findings and user requirements
7. **Secure Execution**: Runs generated code in isolated Modal sandbox environments
8. **Results Synthesis**: Provides detailed summaries with execution results and citations

## Features

### Core Capabilities
- **MCP Server Architecture**: Built on Gradio's MCP server for seamless agent communication
- **Multi-Agent Orchestration**: Sophisticated workflow coordination between specialized agents
- **Real-time Web Research**: Integration with Tavily API for current information retrieval
- **Multi-Provider LLM Support**: Compatible with Nebius, OpenAI, Anthropic, and HuggingFace
- **Intelligent Code Generation**: Context-aware Python code creation with execution capabilities
- **Secure Sandbox Execution**: Modal-based isolated code execution environment
- **Automatic Citation Management**: APA-style citation extraction and formatting

### Advanced Features
- **Performance Monitoring**: Real-time metrics collection and analysis
- **Intelligent Caching**: Advanced caching system for improved response times
- **Circuit Breaker Protection**: Fault-tolerant service protection mechanisms
- **Rate Limiting**: Built-in API rate limiting and request management
- **Health Monitoring**: Comprehensive system health and status tracking
- **Sandbox Pool Management**: Warm sandbox pool for optimized execution performance
- **Async Processing**: High-performance asynchronous operation support

## Prerequisites

- **Python**: Version 3.12 or higher
- **API Keys** for one or more of the following services:
  - **LLM Provider** (choose one):
    - Nebius API (recommended, default)
    - OpenAI API
    - Anthropic API
    - HuggingFace Inference API
  - **Web Search**: Tavily API (required)
  - **Code Execution**: Modal account (required)

## Installation

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mcp_hub_project
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   # Option 1: Using pip with requirements.txt
   pip install -r requirements.txt
   
   # Option 2: Using pyproject.toml (recommended)
   pip install -e .
   
   # Option 3: Manual installation
   pip install gradio[mcp] openai anthropic tavily-python python-dotenv modal psutil aiohttp huggingface-hub
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   # LLM Provider Selection
   LLM_PROVIDER=nebius  # Options: "nebius", "openai", "anthropic", "huggingface"
   
   # API Keys (only set keys for your chosen provider)
   NEBIUS_API_KEY=nb-your-key-here
   # OPENAI_API_KEY=sk-your-key-here
   # ANTHROPIC_API_KEY=sk-ant-your-key-here
   # HUGGINGFACE_API_KEY=hf_your-key-here
   
   # Required for all setups
   TAVILY_API_KEY=tvly-your-key-here
   
   # Optional settings
   CURRENT_YEAR=2025
   ```

## Usage

### Starting the Application

Run the main application:
```bash
python app.py
```

The application will start with:
- **Web Interface**: http://127.0.0.1:7860/
- **MCP Server Schema**: http://127.0.0.1:7860/gradio_api/mcp/schema

### Available Interfaces

The application provides multiple interfaces accessible through tabs:

1. **Orchestrator Flow**: Complete end-to-end research and code generation workflow
2. **Question Enhancer**: Breaks down complex queries into focused sub-questions
3. **Web Search Agent**: Performs targeted web searches using Tavily
4. **LLM Processor**: Text processing with summarization, reasoning, and keyword extraction
5. **Citation Formatter**: Extracts and formats APA-style citations
6. **Code Generator**: Creates Python code based on context and requirements
7. **Code Runner**: Executes Python code in secure Modal sandboxes
8. **Advanced Features**: Health monitoring, performance metrics, and system status

### Example Workflow

1. **Input**: "Create a Python script to analyze Twitter sentiment data"
2. **Question Enhancement**: Breaks into sub-questions about Twitter APIs, sentiment analysis methods, and Python libraries
3. **Web Research**: Searches for current information on each sub-question
4. **Context Integration**: Combines findings into comprehensive research context
5. **Code Generation**: Creates executable Python code with proper imports and structure
6. **Execution**: Runs code in secure sandbox environment
7. **Results**: Returns code, execution output, research summary, and citations

## LLM Provider Configuration

The application supports multiple LLM providers. Set the `LLM_PROVIDER` environment variable to choose your provider:

## Configuration

### LLM Provider Selection

Set the `LLM_PROVIDER` environment variable to configure your preferred provider:

#### Nebius (Default, Recommended)
- **Models**: Qwen and Meta-Llama series
- **API**: OpenAI-compatible interface
- **Advantages**: Cost-effective, high performance, latest models
- **Use Case**: General-purpose research and code generation

#### OpenAI
- **Models**: GPT-4o, GPT-4o-mini, GPT-3.5-turbo
- **API**: Official OpenAI API
- **Advantages**: Proven reliability, excellent reasoning capabilities
- **Use Case**: Complex analysis requiring sophisticated reasoning

#### Anthropic
- **Models**: Claude 3.5 Sonnet, Claude 3 Haiku
- **API**: Official Anthropic API
- **Advantages**: Strong code generation, detailed explanations
- **Use Case**: Code generation and detailed technical documentation

#### HuggingFace Inference
- **Models**: Meta-Llama, StarCoder, and other open models
- **API**: HuggingFace Inference API
- **Advantages**: Open-source models, custom deployments
- **Use Case**: Research projects and custom model experimentation

### Model Selection Strategy

The application automatically selects optimal models based on the task:
- **Question Enhancement**: General language models (GPT-4o-mini, Claude Haiku, Qwen)
- **Summarization**: Efficient models optimized for text processing
- **Code Generation**: Code-specialized models (GPT-4o, Claude Sonnet, Qwen-Coder)
- **Research Tasks**: Balanced models for comprehensive analysis

### Advanced Configuration

Optional settings in `.env`:
```bash
# Performance tuning
NEBIUS_BASE_URL=https://api.studio.nebius.com/v1/
HUGGINGFACE_BASE_URL=https://api-inference.huggingface.co

# Citation formatting
CURRENT_YEAR=2025

# Cache settings (when advanced features enabled)
CACHE_TTL=300
RATE_LIMIT_REQUESTS=100
CIRCUIT_BREAKER_THRESHOLD=5
```

## Architecture & Components

### Agent System
The application consists of specialized agents that work together:

- **Question Enhancer**: Analyzes and breaks down complex queries
- **Web Search Agent**: Performs intelligent web searches via Tavily API
- **LLM Processor**: Handles text processing, summarization, and analysis
- **Citation Formatter**: Manages academic citation formatting
- **Code Generator**: Creates contextually-aware Python code
- **Code Runner**: Executes code in secure Modal sandboxes
- **Orchestrator**: Coordinates the complete workflow between agents

### Core Modules

- **`app.py`**: Main application with Gradio interface and MCP server
- **`mcp_hub/config.py`**: Configuration management for APIs and models
- **`mcp_hub/utils.py`**: Core utility functions and LLM client management
- **`mcp_hub/exceptions.py`**: Custom exception handling
- **`mcp_hub/logging_config.py`**: Logging configuration and management

### Advanced Features (Optional)

- **`mcp_hub/performance_monitoring.py`**: Metrics collection and analysis
- **`mcp_hub/cache_utils.py`**: Intelligent caching system
- **`mcp_hub/reliability_utils.py`**: Circuit breakers and rate limiting
- **`mcp_hub/health_monitoring.py`**: System health monitoring
- **`mcp_hub/sandbox_pool.py`**: Warm sandbox pool management

### MCP Implementation

This project demonstrates advanced MCP (Model Context Protocol) concepts:
- **Server Setup**: Gradio app configured as MCP server (`mcp_server=True`)
- **Function Registration**: Proper MCP function definitions with typing and documentation
- **Multi-Agent Communication**: Structured data flow between specialized agents
- **Schema Generation**: Automatic API schema generation for MCP clients
- **Error Handling**: Robust error management across agent interactions

## Development & Extension

### Project Structure
```
mcp_hub_project/
├── app.py                 # Main application and Gradio interface
├── mcp_hub/              # Core package modules
│   ├── config.py         # Configuration management
│   ├── utils.py          # Core utilities and LLM clients
│   ├── exceptions.py     # Custom exception classes
│   ├── logging_config.py # Logging setup
│   ├── performance_monitoring.py  # Metrics collection
│   ├── cache_utils.py    # Caching system
│   ├── reliability_utils.py       # Circuit breakers, rate limiting
│   ├── health_monitoring.py       # System health monitoring
│   └── sandbox_pool.py   # Sandbox pool management
├── static/               # Static assets (images, CSS)
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Project configuration
└── .env                  # Environment configuration
```

### Adding New Agents

To create a new agent:

1. **Define the agent function** in `app.py`:
   ```python
   @track_performance("my_new_agent")
   def agent_my_new_feature(input_text: str) -> Dict[str, Any]:
       """New agent that does something useful."""
       # Implementation here
       return {"result": "processed_output"}
   ```

2. **Add to Gradio interface**:
   ```python
   with gr.Tab("Agent: My New Feature", scale=1):
       gr.Interface(
           fn=agent_my_new_feature,
           inputs=[gr.Textbox(label="Input")],
           outputs=gr.JSON(label="Output"),
           api_name="agent_my_new_feature_service",
       )
   ```

### Extending LLM Support

To add new LLM providers:

1. **Update `config.py`** with new provider settings
2. **Extend `utils.py`** with client creation logic
3. **Add model selection** in the model configuration

### Performance Optimization

The application includes several optimization features:
- **Intelligent Caching**: Reduces redundant API calls
- **Sandbox Pool**: Pre-warmed execution environments
- **Circuit Breakers**: Prevents cascade failures
- **Rate Limiting**: Manages API usage efficiently
- **Async Processing**: Concurrent operation support

### Monitoring & Debugging

Built-in monitoring capabilities:
- **Performance Metrics**: Track response times and success rates
- **Health Checks**: Monitor system status and dependencies
- **Logging**: Comprehensive logging with configurable levels
- **Cache Analytics**: Monitor cache hit rates and efficiency

## API Reference

### MCP Server Endpoints

When running as an MCP server, the following endpoints are available:

- **`agent_question_enhancer_service`**: Enhance and split questions
- **`agent_web_search_service`**: Perform web searches
- **`agent_llm_processor_service`**: Process text with LLM
- **`agent_citation_formatter_service`**: Format citations
- **`agent_code_generator_service`**: Generate Python code
- **`agent_code_runner_service`**: Execute code in sandbox
- **`process_orchestrator_request`**: Full workflow orchestration

### Advanced Endpoints

- **`get_health_status_service`**: System health information
- **`get_performance_metrics_service`**: Performance analytics
- **`get_cache_status_service`**: Cache statistics
- **`get_sandbox_pool_status_service`**: Sandbox pool metrics

### Response Formats

All agents return structured JSON responses with standardized formats:

```json
{
  "success": true,
  "result": "...",
  "metadata": {
    "timestamp": "2025-06-08T...",
    "processing_time": 1.23,
    "model_used": "gpt-4o-mini"
  },
  "error": null
}
```

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify your API keys are correctly set in `.env`
   - Ensure the selected provider matches your available keys

2. **Import Errors**
   - Install missing dependencies: `pip install -r requirements.txt`
   - For advanced features: `pip install psutil aiohttp`

3. **Modal Sandbox Issues**
   - Ensure Modal is properly configured: `modal setup`
   - Check Modal account status and authentication

4. **Performance Issues**
   - Enable caching for repeated requests
   - Monitor sandbox pool status
   - Check network connectivity for API calls

### Logging

Enable detailed logging by setting environment variables:
```bash
LOG_LEVEL=DEBUG
ENABLE_PERFORMANCE_LOGGING=true
```

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Install development dependencies**: `pip install -e .[dev]`
3. **Follow code style**: Use black and isort for formatting
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Submit a pull request** with clear description

### Development Setup

```bash
# Clone and install in development mode
git clone <repository-url>
cd mcp_hub_project
pip install -e .[dev]

# Run tests
pytest tests/

# Format code
black .
isort .

# Type checking
mypy mcp_hub/
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- **Gradio Team**: For providing excellent MCP server capabilities
- **Nebius**: For high-performance, cost-effective LLM access
- **Tavily**: For reliable web search API integration
- **Modal**: For secure and scalable code execution infrastructure
- **Open Source Community**: For the foundational libraries and tools

---

**Version**: 0.2.0  
**Python Compatibility**: 3.12+  
**Last Updated**: June 2025