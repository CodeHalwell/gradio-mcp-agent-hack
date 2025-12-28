"""
Agent modules for the MCP Hub.

This package contains all specialized agents for the MCP Hub system:
- QuestionEnhancerAgent: Breaks down queries into sub-questions
- WebSearchAgent: Performs web searches using Tavily
- LLMProcessorAgent: Processes text with LLMs
- CitationFormatterAgent: Formats citations from URLs
- CodeGeneratorAgent: Generates secure Python code
- CodeRunnerAgent: Executes code in Modal sandboxes
- OrchestratorAgent: Coordinates all agents
"""

from mcp_hub.agents.question_enhancer import QuestionEnhancerAgent
from mcp_hub.agents.web_search import WebSearchAgent
from mcp_hub.agents.llm_processor import LLMProcessorAgent
from mcp_hub.agents.citation_formatter import CitationFormatterAgent
from mcp_hub.agents.code_generator import CodeGeneratorAgent
from mcp_hub.agents.code_runner import CodeRunnerAgent
from mcp_hub.agents.orchestrator import OrchestratorAgent

__all__ = [
    "QuestionEnhancerAgent",
    "WebSearchAgent",
    "LLMProcessorAgent",
    "CitationFormatterAgent",
    "CodeGeneratorAgent",
    "CodeRunnerAgent",
    "OrchestratorAgent",
]
"""Agent classes for the MCP Hub application."""

from .question_enhancer import QuestionEnhancerAgent
from .web_search import WebSearchAgent
from .llm_processor import LLMProcessorAgent
from .citation_formatter import CitationFormatterAgent
from .code_generator import CodeGeneratorAgent
from .code_runner import CodeRunnerAgent
from .orchestrator import OrchestratorAgent

__all__ = [
    "QuestionEnhancerAgent",
    "WebSearchAgent",
    "LLMProcessorAgent",
    "CitationFormatterAgent",
    "CodeGeneratorAgent",
    "CodeRunnerAgent",
    "OrchestratorAgent",
]
