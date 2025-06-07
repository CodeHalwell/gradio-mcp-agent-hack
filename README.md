# MCP Hub Project - Deep Research & Code Assistant

## Overview

The MCP (Model Context Protocol) Hub is a sophisticated research and code assistant built using Gradio's MCP server functionality. This project demonstrates how to build a workflow of connected AI agents that work together to provide deep research capabilities and generate executable Python code.

The system orchestrates an 8-step deep research and code generation workflow:

1. **Question Enhancement**: Breaks down a user's original query into three distinct sub-questions
2. **Web Search**: Conducts web searches for each sub-question using Tavily API
3. **LLM Summarization**: Summarizes search results for each sub-question using Nebius LLMs
4. **Citation Formatting**: Extracts and formats citations from web search results
5. **Result Combination**: Merges all summaries into a comprehensive grounded context
6. **Code Generation**: Creates Python code based on the research findings using Qwen2.5-Coder-32B-Instruct-fast
7. **Code Execution**: Runs the generated code in a Modal sandbox environment
8. **Final Summary**: Provides a natural language summary of the entire process

## Features

- **MCP Server Implementation**: Built on Gradio's MCP server capabilities for seamless agent communication
- **Multi-Agent Architecture**: Demonstrates how to build interconnected agent services
- **Real-time Web Search**: Integration with Tavily API for up-to-date information
- **Flexible LLM Support**: Compatible with Nebius, OpenAI, Anthropic, and HuggingFace providers
- **Structured Workflow**: Showcases a sophisticated multi-step AI research process
- **Citation Generation**: Automatically formats APA-style citations from web sources
- **Code Generation**: Creates executable Python code based on research findings
- **Code Execution**: Runs generated code in a Modal sandbox environment
- **Final Summary**: Provides a natural language summary of the entire process

## Prerequisites

- Python 3.12+
- API keys for:
  - **One of the following LLM providers**:
    - Nebius API (default)
    - OpenAI API
    - Anthropic API
    - HuggingFace Inference API
  - Tavily API (required for web search)
- Modal account (for code execution in sandbox)

## Installation

1. Clone this repository
2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Activate the virtual environment:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install gradio[mcp] openai anthropic tavily-python python-dotenv modal
# or use the pyproject.toml with your preferred Python package manager:
# pip install -e .
```

4. Create a `.env` file with the following content:
```bash
# Choose your LLM provider: "nebius", "openai", "anthropic", or "huggingface"
LLM_PROVIDER=nebius

# Nebius API (if using nebius provider)
NEBIUS_API_KEY=nb-...

# OpenAI API (if using openai provider)
# OPENAI_API_KEY=sk-...

# Anthropic API (if using anthropic provider)
# ANTHROPIC_API_KEY=sk-ant-...

# HuggingFace API (if using huggingface provider)
# HUGGINGFACE_API_KEY=hf_...

# Required for all providers
TAVILY_API_KEY=tvly-...
CURRENT_YEAR=2025  # Optional, used for citation formatting
```

**Note**: Only set the API key for your chosen provider. See `.env.example` for a complete template.

## Usage

Run the main application:
```bash
python main.py
```

This will launch the Gradio interface at http://127.0.0.1:7860/

The MCP schema will be available at http://127.0.0.1:7860/gradio_api/mcp/schema

## LLM Provider Configuration

The application supports multiple LLM providers. Set the `LLM_PROVIDER` environment variable to choose your provider:

### Nebius (Default)
- Models: Qwen and Meta-Llama models
- API: OpenAI-compatible interface
- Best for: General-purpose tasks with good performance/cost ratio

### OpenAI
- Models: GPT-4o, GPT-4o-mini
- API: Official OpenAI API
- Best for: High-quality responses and complex reasoning

### Anthropic
- Models: Claude 3.5 Sonnet, Claude 3 Haiku
- API: Official Anthropic API
- Best for: Detailed analysis and code generation

### HuggingFace Inference
- Models: Meta-Llama, StarCoder, and other open models
- API: HuggingFace Inference API
- Best for: Open-source models and custom deployments

The application automatically selects appropriate models for each task based on the chosen provider.

## Available Agents

The project includes several agent services:

1. **Question Enhancer**: Splits a request into three sub-questions using the configured LLM provider
2. **Web Search Agent**: Performs web searches via Tavily API (top-3 results)
3. **LLM Processor**: Processes text with the configured LLM provider for summarization, reasoning, or keyword extraction
4. **Citation Formatter**: Extracts URLs and formats them as APA-style citations
5. **Code Generator**: Creates Python code snippets based on research context using the configured LLM provider
6. **Code Runner**: Executes Python code in a Modal sandbox environment
7. **Orchestrator**: Coordinates all agents in a cohesive workflow

## Tutorial Scripts

The `tutorial_scripts/` directory contains example Gradio applications and code samples for learning:

- `simple_app.py`: A basic Gradio interface
- `letter_count.py`: A simple letter counting example
- `predict_letter_count.py`: Example of letter counting prediction
- `modal_inference.py`: Demonstrates using Modal for code execution
- `nebius_inference.py`: Shows how to use Nebius API for inference
- `nebius_tool_calling.py`: Example of tool calling with Nebius models
- `Gradio Cheat Sheet.md`: Quick reference for Gradio features and usage

## MCP Implementation Details

This project demonstrates how to:
- Create MCP-compatible function definitions with proper typing and docstrings
- Launch a Gradio app as an MCP server (`mcp_server=True`)
- Structure a multi-agent workflow
- Pass data between agents in a structured format
- Execute code safely in a sandbox environment

## Example Workflow

1. A user submits a high-level request like "Write Python code to analyze sentiment from Twitter data"
2. The system breaks this into three sub-questions (e.g., about Twitter APIs, sentiment analysis techniques, and Python libraries)
3. For each sub-question, it:
   - Performs a web search using Tavily
   - Summarizes the search results
   - Extracts citations from URLs
4. The sub-summaries are combined into a comprehensive grounded context
5. Based on this context, Python code is generated
6. The code is executed in a Modal sandbox
7. The user receives the final summary, generated code, execution output, and citations

## License

[Your license information here]

## Contributing

[Your contribution guidelines here]

## Acknowledgments

- Gradio for providing the MCP server functionality
- Nebius for LLM capabilities
- Tavily for web search capabilities