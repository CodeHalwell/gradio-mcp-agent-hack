---
title: ShallowCodeResearch
emoji: 📉
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: 5.33.1
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
- mcp-server-track
python_version: '3.12'
---

# Shallow Research Code Assistant - Multi-Agent AI Code Assistant

## Technologies Used

This is part of the MCP track for the Hackathon (with a smidge of Agents)

- Gradio for the UI and MCP logic
- Modal AI for spinning up sandboxes for code execution
- Nebius, OpenAI, Anthropic and Hugging Face can be used for LLM calls
- Nebius set by default for inference, with a priority on token speed that can be found on the platform

❤️ **A very big thank you to the sponsors for the generous credits for this hackathon and Hugging Face and Gradio for putting this event together** 🔥

<a href="https://glama.ai/mcp/servers/@CodeHalwell/gradio-mcp-agent-hack">
  <img width="380" height="200" src="https://glama.ai/mcp/servers/@CodeHalwell/gradio-mcp-agent-hack/badge" alt="Hub MCP server" />
</a>

**Special thanks to Yuvi for putting up with us in the Discord asking for credits 😂**

## 🚀 **Multi-agent system for AI-powered search and code generation**

## What is the Shallow Research MCP Hub for Code Assistance?

Shallow Research Code Assistant is a sophisticated multi-agent research and code assistant built using Gradio's Model Context Protocol (MCP) server functionality. It orchestrates specialized AI agents to provide comprehensive research capabilities and generate executable Python code. This "shallow" research tool (Its definitely not deep research) augments
the initial user query to broaden scope before performing web searches for grounding.

The coding agent then generates the code to answer the user question and checks for errors. To ensure the code is valid, the code is executed in a remote sandbox using the
Modal infrustructure. These sandboxes are spawned when needed with a small footprint (only pandas, numpy, request and scikit-learn are installed).

However, if additional packages are required, this will be installed prior to execution (some delays expected here depending on the request).

Once executed the whole process is summarised and returned to the user.

---
## 📹 Demo Video

[![MCP Demo Shallow Research Code Assistant](https://img.shields.io/badge/Watch%20Demo-Loom-purple?style=for-the-badge&logo=loom)](https://www.loom.com/share/ea4584bc76c04adabefd6d39a4f8e279?sid=5d2408ff-03d1-421b-b956-9713ae390212)

*Click the badge above to watch the complete demonstration of the MCP Demo Shallow Research Code Assistant in action*

---

## Key information

I've found that whilst using VS Code for the MCP interaction, its useful to type the main agent function name to ensure the right tool is picked.

For example "agent research request: How do you write a python script to perform scaling of features in a dataframe"

This is the JSON script required to set up the MCP in VS Code

```json
{
    "mcp": {
        "inputs": [],
        "servers": {
        "gradiocodeassist": {
            "command": "npx",
            "args": [
                "mcp-remote",
                "https://agents-mcp-hackathon-shallowcoderesearch.hf.space/gradio_api/mcp/sse",
            ]
        }
    }
}
```
This is the JSON script required to set up the MCP Via Cline in VS Code

```json
{
  "mcpServers": {
    "gradiocodeassist": {
      "autoApprove": [],
      "disabled": false,
      "timeout": 300,
      "type": "sse",
      "url": "https://agents-mcp-hackathon-shallowcoderesearch.hf.space/gradio_api/mcp/sse",
      "headers": {}
    }
  }
}
```

## ✨ Key Features

- 🧠 **Multi-Agent Architecture**: Specialized agents working in orchestrated workflows
- 🔍 **Intelligent Research**: Web search with automatic summarization and citation formatting
- 💻 **Code Generation**: Context-aware Python code creation with secure execution
- 🔗 **MCP Server**: Built-in MCP server for seamless agent communication
- 🎯 **Multiple LLM Support**: Compatible with Nebius, OpenAI, Anthropic, and HuggingFace (Currently set to Nebius Inference)
- 🛡️ **Secure Execution**: Modal sandbox environment for safe code execution
- 📊 **Performance Monitoring**: Advanced metrics collection and health monitoring

## 🏛️ MCP Workflow Architecture

![MCP Workflow Diagram](static/MCP_Diagram.png)

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

## 📝 License

This project is licensed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute this software with proper attribution. See the `LICENSE` file for details.
