"""Utility functions for the MCP Hub project."""

import json
import re
from typing import Dict, Any, List, Optional, Union
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

def create_llm_client() -> Union[OpenAI, object]:
    """Create and return an LLM client based on the configured provider."""
    if api_config.llm_provider == "nebius":
        return create_nebius_client()
    elif api_config.llm_provider == "openai":
        return OpenAI(api_key=api_config.openai_api_key)
    elif api_config.llm_provider == "anthropic":
        try:
            import anthropic
            return anthropic.Anthropic(api_key=api_config.anthropic_api_key)
        except ImportError:
            raise APIError("Anthropic", "anthropic package not installed. Install with: pip install anthropic")
    elif api_config.llm_provider == "huggingface":
        # HuggingFace uses requests/aiohttp for inference
        return None  # Will be handled separately in make_completion functions
    else:
        raise APIError("Config", f"Unsupported LLM provider: {api_config.llm_provider}")

def create_async_llm_client() -> Union[AsyncOpenAI, object]:
    """Create and return an async LLM client based on the configured provider."""
    if api_config.llm_provider == "nebius":
        return create_async_nebius_client()
    elif api_config.llm_provider == "openai":
        return AsyncOpenAI(api_key=api_config.openai_api_key)
    elif api_config.llm_provider == "anthropic":
        try:
            import anthropic
            return anthropic.AsyncAnthropic(api_key=api_config.anthropic_api_key)
        except ImportError:
            raise APIError("Anthropic", "anthropic package not installed. Install with: pip install anthropic")
    elif api_config.llm_provider == "huggingface":
        # HuggingFace uses requests/aiohttp for inference
        return None  # Will be handled separately in make_completion functions
    else:
        raise APIError("Config", f"Unsupported LLM provider: {api_config.llm_provider}")

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

def make_llm_completion(
    model: str,
    messages: List[Dict[str, str]], 
    temperature: float = 0.6,
    response_format: Optional[Dict[str, Any]] = None
) -> str:
    """Make a completion request using the configured LLM provider."""
    provider = api_config.llm_provider
    
    try:
        if provider == "nebius":
            return make_nebius_completion(model, messages, temperature, response_format)
        
        elif provider == "openai":
            client = create_llm_client()
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if response_format:
                kwargs["response_format"] = response_format
            completion = client.chat.completions.create(**kwargs)
            return completion.choices[0].message.content.strip()
        
        elif provider == "anthropic":
            client = create_llm_client()
            # Convert OpenAI format to Anthropic format
            anthropic_messages = []
            system_message = None
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            kwargs = {
                "model": model,
                "messages": anthropic_messages,
                "temperature": temperature,
                "max_tokens": 1000,
            }
            if system_message:
                kwargs["system"] = system_message
            
            response = client.messages.create(**kwargs)
            return response.content[0].text.strip()
        
        elif provider == "huggingface":
            import requests
            
            headers = {
                "Authorization": f"Bearer {api_config.huggingface_api_key}",
                "Content-Type": "application/json"
            }
            
            # Convert messages to a simple prompt for HF inference
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n"
                elif msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"
            prompt += "Assistant:"
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": 1000,
                    "return_full_text": False
                }
            }
            
            response = requests.post(
                f"{api_config.huggingface_base_url}/models/{model}",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                raise APIError("HuggingFace", f"HTTP {response.status_code}: {response.text}")
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").strip()
            else:
                raise APIError("HuggingFace", "Unexpected response format")
        
        else:
            raise APIError("Config", f"Unsupported provider: {provider}")
            
    except Exception as e:
        if isinstance(e, APIError):
            raise
        raise APIError(provider.title(), f"Completion failed: {str(e)}")

async def make_async_llm_completion(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    response_format: Optional[Dict[str, Any]] = None,
) -> str:
    """Make an async completion request using the configured LLM provider."""
    provider = api_config.llm_provider
    
    try:
        if provider == "nebius":
            return await make_async_nebius_completion(model, messages, temperature, response_format)
        
        elif provider == "openai":
            client = create_async_llm_client()
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
            if response_format:
                kwargs["response_format"] = response_format
            
            response = await client.chat.completions.create(**kwargs)
            
            if not response.choices:
                raise APIError("OpenAI", "No completion choices returned")
            
            content = response.choices[0].message.content
            if content is None:
                raise APIError("OpenAI", "Empty response content")
            
            return content.strip()
        
        elif provider == "anthropic":
            client = create_async_llm_client()
            # Convert OpenAI format to Anthropic format
            anthropic_messages = []
            system_message = None
            
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            kwargs = {
                "model": model,
                "messages": anthropic_messages,
                "temperature": temperature,
                "max_tokens": 1000,
            }
            if system_message:
                kwargs["system"] = system_message
            
            response = await client.messages.create(**kwargs)
            return response.content[0].text.strip()
        
        elif provider == "huggingface":
            headers = {
                "Authorization": f"Bearer {api_config.huggingface_api_key}",
                "Content-Type": "application/json"
            }
            
            # Convert messages to a simple prompt for HF inference
            prompt = ""
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"System: {msg['content']}\n"
                elif msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"
            prompt += "Assistant:"
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": 1000,
                    "return_full_text": False
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{api_config.huggingface_base_url}/models/{model}",
                    headers=headers,
                    json=payload,
                    timeout=30
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise APIError("HuggingFace", f"HTTP {response.status}: {text}")
                    
                    result = await response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get("generated_text", "").strip()
                    else:
                        raise APIError("HuggingFace", "Unexpected response format")
        
        else:
            raise APIError("Config", f"Unsupported provider: {provider}")
            
    except Exception as e:
        if isinstance(e, APIError):
            raise
        raise APIError(provider.title(), f"Async completion failed: {str(e)}")

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
