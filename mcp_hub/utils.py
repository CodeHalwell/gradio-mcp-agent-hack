"""Utility functions for the MCP Hub project."""

import json
import re
from typing import Dict, Any, List, Optional
from openai import OpenAI, AsyncOpenAI
from .config import api_config
from .exceptions import APIError, ValidationError
import asyncio
import aiohttp

def create_nebius_client() -> OpenAI:
    """Create and return a Nebius OpenAI client."""
    return OpenAI(
        base_url=api_config.nebius_base_url,
        api_key=api_config.nebius_api_key,
    )

def create_async_nebius_client() -> AsyncOpenAI:
    """Create and return an async Nebius OpenAI client."""
    return AsyncOpenAI(
        base_url=api_config.nebius_base_url,
        api_key=api_config.nebius_api_key,
    )

def validate_non_empty_string(value: str, field_name: str) -> None:
    """Validate that a string is not empty or None."""
    if not value or not value.strip():
        raise ValidationError(f"{field_name} cannot be empty.")

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON object from text that may contain markdown fences."""
    # Remove markdown code fences if present
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()
        else:
            text = text.strip("```").strip()
    
    # Find JSON object boundaries
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        raise ValidationError("Failed to locate JSON object in text.")
    
    json_candidate = text[start_idx:end_idx + 1]
    
    try:
        return json.loads(json_candidate)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Failed to parse JSON: {str(e)}")

def extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from text using regex."""
    url_pattern = r"(https?://[^\s]+)"
    return re.findall(url_pattern, text)

def make_nebius_completion(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.6,
    response_format: Optional[Dict[str, Any]] = None
) -> str:
    """Make a completion request to Nebius and return the content."""
    client = create_nebius_client()
    
    try:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        completion = client.chat.completions.create(**kwargs)
        return completion.choices[0].message.content.strip()
    except Exception as e:
        raise APIError("Nebius", str(e))

async def make_async_nebius_completion(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    response_format: Optional[Dict[str, Any]] = None,
) -> str:
    """Make an async completion request to Nebius API."""
    try:
        client = create_async_nebius_client()
        
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        
        response = await client.chat.completions.create(**kwargs)
        
        if not response.choices:
            raise APIError("Nebius", "No completion choices returned")
        
        content = response.choices[0].message.content
        if content is None:
            raise APIError("Nebius", "Empty response content")
        
        return content.strip()
        
    except Exception as e:
        if isinstance(e, APIError):
            raise
        raise APIError("Nebius", f"API call failed: {str(e)}")

async def async_tavily_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """Perform async web search using Tavily API."""
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://api.tavily.com/search"
            headers = {
                "Content-Type": "application/json"
            }
            data = {
                "api_key": api_config.tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": True
            }
            
            async with session.post(url, headers=headers, json=data) as response:
                if response.status != 200:
                    raise APIError("Tavily", f"HTTP {response.status}: {await response.text()}")
                
                result = await response.json()
                return {
                    "query": result.get("query", query),
                    "tavily_answer": result.get("answer"),
                    "results": result.get("results", []),
                    "data_source": "Tavily Search API",
                }
                
    except aiohttp.ClientError as e:
        raise APIError("Tavily", f"HTTP request failed: {str(e)}")
    except Exception as e:
        if isinstance(e, APIError):
            raise
        raise APIError("Tavily", f"Search failed: {str(e)}")

def format_search_results(results: List[Dict[str, Any]]) -> str:
    """Format search results into a readable string."""
    if not results:
        return "No search results found."
    
    snippets = []
    for idx, item in enumerate(results, 1):
        title = item.get("title", "No Title")
        url = item.get("url", "")
        content = item.get("content", "")
        
        snippet = f"Result {idx}:\nTitle: {title}\nURL: {url}\nSnippet: {content}\n"
        snippets.append(snippet)
    
    return "\n".join(snippets).strip()

def create_apa_citation(url: str, year: str = None) -> str:
    """Create a simple APA-style citation from a URL."""
    if not year:
        year = api_config.current_year
    
    try:
        domain = url.split("/")[2]
        title = domain.replace("www.", "").split(".")[0].capitalize()
        return f"{title}. ({year}). Retrieved from {url}"
    except (IndexError, AttributeError):
        return f"Unknown Source. ({year}). Retrieved from {url}"
