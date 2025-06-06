# MCP Hub Project

## Overview

The MCP (Model Context Protocol) Hub is a sophisticated research assistant built using Gradio's MCP server functionality. This project demonstrates how to build a workflow of connected AI agents that work together to provide deep research capabilities.

The system orchestrates a 5-step deep research workflow:

1. **Question Enhancement**: Breaks down a user's original query into five distinct sub-questions
2. **Web Search**: Conducts web searches for each sub-question using Tavily API
3. **LLM Summarization**: Summarizes search results for each sub-question using Nebius LLMs
4. **Citation Formatting**: Extracts and formats citations from web search results
5. **Result Combination**: Merges all summaries into a comprehensive final answer

## Features

- **MCP Server Implementation**: Built on Gradio's MCP server capabilities for seamless agent communication
- **Multi-Agent Architecture**: Demonstrates how to build interconnected agent services
- **Real-time Web Search**: Integration with Tavily API for up-to-date information
- **LLM Processing**: Uses Nebius (OpenAI-compatible) models for text processing
- **Structured Workflow**: Showcases a sophisticated multi-step AI research process
- **Citation Generation**: Automatically formats APA-style citations from web sources

## Prerequisites

- Python 3.12+
- API keys for:
  - Nebius API
  - Tavily API

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
pip install -r requirements.txt
# or use the pyproject.toml with your preferred Python package manager
```

4. Create a `.env` file with the following content:
```
NEBIUS_API_KEY=nb-...
TAVILY_API_KEY=tvly-...
```

## Usage

Run the main application:
```bash
python main.py
```

This will launch the Gradio interface at http://127.0.0.1:7860/

The MCP schema will be available at http://127.0.0.1:7860/gradio_api/mcp/schema

## Available Agents

The project includes several agent services:

1. **Question Enhancer**: Splits a request into five sub-questions using Qwen3-32B-fast
2. **Web Search Agent**: Performs web searches via Tavily API
3. **LLM Processor**: Processes text with Nebius LLMs for summarization, reasoning, or keyword extraction
4. **Citation Formatter**: Extracts URLs and formats them as APA-style citations
5. **Orchestrator**: Coordinates all agents in a cohesive workflow

## Tutorial Scripts

The `tutorial_scripts/` directory contains example Gradio applications for learning:

- `simple_app.py`: A basic Gradio interface
- `letter_count.py`: A simple letter counting MCP server
- And more examples demonstrating various Gradio and MCP features

## MCP Implementation Details

This project demonstrates how to:
- Create MCP-compatible function definitions with proper typing and docstrings
- Launch a Gradio app as an MCP server (`mcp_server=True`)
- Structure a multi-agent workflow
- Pass data between agents in a structured format

## License

[Your license information here]

## Contributing

[Your contribution guidelines here]

## Acknowledgments

- Gradio for providing the MCP server functionality
- Nebius for LLM capabilities
- Tavily for web search capabilities