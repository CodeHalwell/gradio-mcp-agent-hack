# MCP Hub Architecture

## Overview

The MCP Hub is a modular, multi-agent system for AI-assisted research, code generation, and execution. The codebase has been refactored for improved maintainability, testability, and clarity.

## Project Structure

```
gradio-mcp-agent-hack/
├── app.py                          # Main application entry point (917 lines)
├── mcp_hub/                        # Core package
│   ├── __init__.py                 # Package initialization with exports
│   ├── agents/                     # Agent implementations
│   │   ├── __init__.py             # Agent exports
│   │   ├── question_enhancer.py   # Question enhancement agent
│   │   ├── web_search.py           # Web search agent
│   │   ├── llm_processor.py        # LLM processing agent
│   │   ├── citation_formatter.py   # Citation formatting agent
│   │   ├── code_generator.py       # Code generation agent
│   │   ├── code_runner.py          # Code execution agent
│   │   └── orchestrator.py         # Main orchestration agent
│   ├── decorators.py               # Performance tracking decorators
│   ├── config.py                   # Configuration management
│   ├── exceptions.py               # Custom exceptions
│   ├── logging_config.py           # Logging setup
│   ├── utils.py                    # Utility functions
│   ├── async_utils.py              # Async utilities
│   ├── cache_utils.py              # Caching utilities
│   ├── reliability_utils.py        # Circuit breakers, rate limiting
│   ├── performance_monitoring.py   # Performance metrics
│   ├── health_monitoring.py        # Health checks
│   ├── sandbox_pool.py             # Sandbox pool management
│   └── package_utils.py            # Package management utilities
├── tests/                          # Test suite
│   ├── unit/                       # Unit tests
│   │   ├── test_question_enhancer_agent.py
│   │   ├── test_web_search_agent.py
│   │   ├── test_llm_processor_agent.py
│   │   ├── test_citation_formatter_agent.py
│   │   └── test_code_generator_agent.py
│   ├── integration/                # Integration tests
│   └── conftest.py                 # Test fixtures
├── requirements.txt                # Project dependencies
└── pyproject.toml                  # Project metadata
```

## Agent Architecture

### Agent Hierarchy

```
OrchestratorAgent (Coordinates all agents)
    ├── QuestionEnhancerAgent (Breaks down user queries)
    ├── WebSearchAgent (Performs web searches)
    ├── LLMProcessorAgent (Processes text with LLMs)
    ├── CitationFormatterAgent (Formats citations)
    ├── CodeGeneratorAgent (Generates Python code)
    └── CodeRunnerAgent (Executes code in sandboxes)
```

### Agent Responsibilities

#### QuestionEnhancerAgent
- **Purpose**: Breaks down complex user queries into manageable sub-questions
- **Key Methods**: `enhance_question(user_request, num_questions)`
- **Features**: Uses LLM to generate distinct, non-overlapping sub-questions

#### WebSearchAgent
- **Purpose**: Performs web searches using Tavily API
- **Key Methods**: `search(query)`, `search_async(query)`
- **Features**: Both sync and async search, result caching, error handling

#### LLMProcessorAgent
- **Purpose**: Processes text using Large Language Models
- **Key Methods**: `process(text, task, context)`, `async_process(text, task, context)`
- **Supported Tasks**: summarization, reasoning, keyword extraction

#### CitationFormatterAgent
- **Purpose**: Extracts URLs and formats APA-style citations
- **Key Methods**: `format_citations(text_block)`
- **Features**: Automatic URL extraction, APA formatting

#### CodeGeneratorAgent
- **Purpose**: Generates secure Python code from user requests
- **Key Methods**: `generate_code(user_request, grounded_context)`
- **Features**: Security checks, syntax validation, iterative error correction

#### CodeRunnerAgent
- **Purpose**: Executes Python code in isolated Modal sandboxes
- **Key Methods**: `run_code(code)`, `run_code_async(code)`
- **Features**: Warm sandbox pools, package management, safety shims

#### OrchestratorAgent
- **Purpose**: Coordinates the complete workflow across all agents
- **Key Methods**: `orchestrate(user_request)`
- **Workflow**: 
  1. Enhance question
  2. Search for information
  3. Create grounded context
  4. Generate code
  5. Execute code
  6. Format citations
  7. Generate summary

## Key Design Patterns

### 1. Decorator Pattern
- Performance tracking decorator wraps all agent methods
- Rate limiting and circuit breaker decorators for API protection
- Caching decorator for performance optimization

### 2. Dependency Injection
- Agents are instantiated once and injected where needed
- Facilitates testing with mock agents
- Reduces coupling between components

### 3. Graceful Degradation
- Advanced features loaded with try/except
- Fallback implementations for missing dependencies
- System continues to function with reduced features

### 4. Async-First Design
- Async methods provided alongside sync methods
- Better performance for concurrent operations
- Event loop management for Gradio compatibility

## Module Descriptions

### Core Modules

#### `mcp_hub/config.py`
- Manages API keys, model configurations, and app settings
- Environment variable loading via `python-dotenv`
- Centralized configuration management

#### `mcp_hub/exceptions.py`
- Custom exception hierarchy for better error handling
- Exceptions: `APIError`, `ValidationError`, `CodeGenerationError`, `CodeExecutionError`

#### `mcp_hub/decorators.py`
- Performance tracking decorator
- Integrates with advanced monitoring when available
- Handles both sync and async functions

#### `mcp_hub/utils.py`
- Utility functions for validation, JSON extraction, LLM completion
- URL extraction and APA citation generation
- Shared helpers used across agents

### Advanced Features

#### `mcp_hub/performance_monitoring.py`
- Metrics collection and aggregation
- Operation timing and success tracking
- Error counting and categorization

#### `mcp_hub/reliability_utils.py`
- Rate limiting for API protection
- Circuit breaker pattern for fault tolerance
- Backoff strategies for retries

#### `mcp_hub/cache_utils.py`
- Caching decorator with TTL support
- Reduces redundant API calls
- Improves response times

#### `mcp_hub/sandbox_pool.py`
- Warm sandbox pool management
- Reduces cold start times for code execution
- Sandbox lifecycle management

## Benefits of Refactoring

### Improved Maintainability
- ✅ Reduced main file size from 2,424 to 917 lines (62% reduction)
- ✅ Each agent class in its own file (average 150-400 lines)
- ✅ Clear separation of concerns
- ✅ Easier to locate and modify specific functionality

### Better Testability
- ✅ Individual agents can be tested in isolation
- ✅ Mock implementations easier to create
- ✅ Test files mirror the module structure
- ✅ Reduced test setup complexity

### Enhanced Extensibility
- ✅ New agents can be added without modifying existing code
- ✅ Clear interfaces for agent implementation
- ✅ Plugin-style architecture for features
- ✅ Easier to add new capabilities

### Clearer Dependencies
- ✅ Import statements reveal dependencies
- ✅ No circular dependencies
- ✅ Easier to understand data flow
- ✅ Better documentation through structure

## Migration Guide

### Before Refactoring
```python
# Everything in app.py
from app import QuestionEnhancerAgent, WebSearchAgent, ...

agent = QuestionEnhancerAgent()
```

### After Refactoring
```python
# Import from organized modules
from mcp_hub.agents import QuestionEnhancerAgent, WebSearchAgent

# Or import all at once
from mcp_hub.agents import (
    QuestionEnhancerAgent,
    WebSearchAgent,
    LLMProcessorAgent,
    CitationFormatterAgent,
    CodeGeneratorAgent,
    CodeRunnerAgent,
    OrchestratorAgent,
)

agent = QuestionEnhancerAgent()
```

### Backward Compatibility

The `app.py` file still instantiates all agents at the module level for backward compatibility:

```python
# These are still available in app.py
question_enhancer = QuestionEnhancerAgent()
web_search = WebSearchAgent()
llm_processor = LLMProcessorAgent()
# ...
```

## Testing

### Unit Tests
```bash
pytest tests/unit/
```

### Integration Tests
```bash
pytest tests/integration/
```

### All Tests
```bash
pytest tests/
```

## Running the Application

### Standard Launch
```bash
python app.py
```

### With MCP Server
```bash
# MCP server is enabled by default in app.py
python app.py
```

## Future Improvements

### Potential Enhancements
1. **UI Module**: Move Gradio UI components to `mcp_hub/ui/`
2. **API Module**: Create REST API endpoints in `mcp_hub/api/`
3. **Database Layer**: Add persistence with `mcp_hub/db/`
4. **Agent Registry**: Dynamic agent discovery and loading
5. **Plugin System**: Allow third-party agents
6. **Distributed Execution**: Support for distributed agent execution

### Technical Debt
- None currently identified (fresh refactoring)

## Contributing

When adding new agents:
1. Create a new file in `mcp_hub/agents/`
2. Implement the agent class with clear docstrings
3. Add exports to `mcp_hub/agents/__init__.py`
4. Create corresponding test file in `tests/unit/`
5. Update this documentation

## Performance Metrics

### Code Organization
- **Before**: 1 file, 2,424 lines
- **After**: 9 agent files, 1,611 total lines + 98 lines decorators + 917 lines main = ~2,626 lines
- **Improvement**: Better organization despite slight line increase (due to imports and docstrings)

### File Sizes
- `app.py`: 917 lines (was 2,424) - **62% reduction**
- Largest agent file: `code_runner.py` (419 lines)
- Smallest agent file: `citation_formatter.py` (60 lines)
- Average agent file size: ~230 lines

## License

[Your License Here]

## Authors

- Original implementation: CodeHalwell
- Refactoring: Copilot Agent

## Changelog

### Version 1.0.0 (Refactored)
- ✅ Separated agent classes into individual modules
- ✅ Created `mcp_hub/agents/` package
- ✅ Extracted performance tracking decorator to `mcp_hub/decorators.py`
- ✅ Updated imports and exports
- ✅ Maintained backward compatibility
- ✅ All files pass syntax validation
