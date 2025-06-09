---
title: ShallowCodeResearch
emoji: 📉
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
short_description: Coding research assistant that generates code and tests it
tags:
  - mcp
  - multi-agent
  - research
  - code-generation
  - ai-assistant
  - gradio
  - python
  - web-search
  - llm
  - modal
python_version: "3.12"
---

# Shallow Research Code Assistant - Multi-Agent AI Code Assistant

🚀 **Multi-agent system for AI-powered search and code generation**

## What is MCP Hub?

Shallow Research Code Assistant is a sophisticated multi-agent research and code assistant built using Gradio's Model Context Protocol (MCP) server functionality. It orchestrates specialized AI agents to provide comprehensive research capabilities and generate executable Python code. This "shallow" research tool (Its definitely not deep research) augments
the initial user query to broaden scope before performing web searches for grounding.

The coding agent then generates the code to answer the user question and checks for errors. To ensure the code is valid, the code is executed in a remote sandbox using the
Modal infrustructure. These sandboxes are spawned when needed with a small footprint (only pandas, numpy, request and scikit-learn are installed).

However, if additional packages are required, this will be installed prior to execution (some delays expected here depending on the request).

Once executed the whole process is summarised and returned to the user.

## ✨ Key Features

- 🧠 **Multi-Agent Architecture**: Specialized agents working in orchestrated workflows
- 🔍 **Intelligent Research**: Web search with automatic summarization and citation formatting
- 💻 **Code Generation**: Context-aware Python code creation with secure execution
- 🔗 **MCP Server**: Built-in MCP server for seamless agent communication
- 🎯 **Multiple LLM Support**: Compatible with Nebius, OpenAI, Anthropic, and HuggingFace (Currently set to Nebius Inference)
- 🛡️ **Secure Execution**: Modal sandbox environment for safe code execution
- 📊 **Performance Monitoring**: Advanced metrics collection and health monitoring

## 🏛️ MCP Workflow Architecture

![MCP Workflow Diagram](MCP%20Diagram.png)

The diagram above illustrates the complete Multi-Agent workflow architecture, showing how different agents communicate through the MCP (Model Context Protocol) server to deliver comprehensive research and code generation capabilities.


## 🚀 Quick Start

1. **Configure your environment** by setting up API keys in the Settings tab
2. **Choose your LLM provider** Nebius Set By Default in the Space
3. **Input your research query** in the Orchestrator Flow tab
4. **Watch the magic happen** as agents collaborate to research and generate code

## 🏗️ Architecture

### Core Agents

- **Question Enhancer**: Breaks down complex queries into focused sub-questions
- **Web Search Agent**: Performs targeted searches using Tavily API
- **LLM Processor**: Handles text processing, summarization, and analysis
- **Citation Formatter**: Manages academic citation formatting (APA style)
- **Code Generator**: Creates contextually-aware Python code
- **Code Runner**: Executes code in secure Modal sandboxes
- **Orchestrator**: Coordinates the complete workflow

### Workflow Example

```
User Query: "Create Python code to analyze Twitter sentiment"
    ↓
Question Enhancement: Split into focused sub-questions
    ↓
Web Research: Search for Twitter APIs, sentiment libraries, examples
    ↓
Context Integration: Combine research into comprehensive context
    ↓
Code Generation: Create executable Python script
    ↓
Secure Execution: Run code in Modal sandbox
    ↓
Results: Code + output + research summary + citations
```

## 🛠️ Setup Requirements

### Required API Keys

- **LLM Provider** (choose one):
  - Nebius API (recommended)
  - OpenAI API
  - Anthropic API
  - HuggingFace Inference API
- **Tavily API** (for web search)
- **Modal Account** (for code execution)

### Environment Configuration

Set these environment variables or configure in the app:

```bash
LLM_PROVIDER=nebius  # Your chosen provider
NEBIUS_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
MODAL_ID=your-id-here
MODEL_SECRET_TOKEN=your-token-here
```

## 🎯 Use Cases

### Code Generation
- **Prototype Development**: Rapidly create functional code based on requirements
- **IDE Integration**: Add this to your IDE for grounded LLM support

### Learning & Education
- **Code Examples**: Generate educational code samples with explanations
- **Concept Exploration**: Research and understand complex programming concepts
- **Best Practices**: Learn current industry standards and methodologies

## 🔧 Advanced Features

### Performance Monitoring
- Real-time metrics collection
- Response time tracking
- Success rate monitoring
- Resource usage analytics

### Intelligent Caching
- Reduces redundant API calls
- Improves response times
- Configurable TTL settings

### Fault Tolerance
- Circuit breaker protection
- Rate limiting management
- Graceful error handling
- Automatic retry mechanisms

### Sandbox Pool Management
- Pre-warmed execution environments
- Optimized performance
- Resource pooling
- Automatic scaling

## 📱 Interface Tabs

1. **Orchestrator Flow**: Complete end-to-end workflow
2. **Individual Agents**: Access each agent separately for specific tasks
3. **Advanced Features**: System monitoring and performance analytics

## 🤝 MCP Integration

This application demonstrates advanced MCP (Model Context Protocol) implementation:

- **Server Architecture**: Full MCP server with schema generation
- **Function Registry**: Proper MCP function definitions with typing
- **Multi-Agent Communication**: Structured data flow between agents
- **Error Handling**: Robust error management across agent interactions

## 📊 Performance

- **Response Times**: Optimized for sub-second agent responses
- **Scalability**: Handles concurrent requests efficiently
- **Reliability**: Built-in fault tolerance and monitoring
- **Resource Management**: Intelligent caching and pooling

## 🔍 Technical Details

- **Python**: 3.12+ required
- **Framework**: Gradio with MCP server capabilities
- **Execution**: Modal for secure sandboxed code execution
- **Search**: Tavily API for real-time web research
- **Monitoring**: Comprehensive performance and health tracking

---

**Ready to experience the future of AI-assisted research and development?** 

Start by configuring your API keys and dive into the world of multi-agent AI collaboration! 🚀
