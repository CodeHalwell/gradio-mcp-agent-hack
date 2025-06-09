"""Utility functions for the MCP Hub project."""

import json
import re
from typing import Dict, Any, List, Optional, Union
from openai import OpenAI, AsyncOpenAI
from .config import api_config, model_config
from .exceptions import APIError, ValidationError
from .logging_config import logger
import aiohttp
from huggingface_hub import InferenceClient


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
        # Try different HuggingFace client configurations for better compatibility
        try:
            # First try with hf-inference provider (most recent approach)
            return InferenceClient(
                provider="hf-inference",
                api_key=api_config.huggingface_api_key,
            )
        except Exception:
            # Fallback to token-based authentication
            return InferenceClient(
                token=api_config.huggingface_api_key,
            )
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
        # Try different HuggingFace client configurations for better compatibility
        try:
            # First try with hf-inference provider (most recent approach)
            return InferenceClient(
                provider="hf-inference",
                api_key=api_config.huggingface_api_key,
            )
        except Exception:
            # Fallback to token-based authentication
            return InferenceClient(
                token=api_config.huggingface_api_key,
            )
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
            # OpenAI only supports simple response_format, not the extended Nebius format
            if response_format and response_format.get("type") == "json_object":
                kwargs["response_format"] = {"type": "json_object"}
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
            # Try HuggingFace with fallback to Nebius
            hf_error = None
            try:
                client = create_llm_client()
                
                # Try multiple HuggingFace API approaches
                
                # Method 1: Try chat.completions.create (OpenAI-compatible)
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=1000,
                    )
                    
                    # Extract the response content
                    if hasattr(response, 'choices') and response.choices:
                        return response.choices[0].message.content.strip()
                    else:
                        return str(response).strip()
                        
                except Exception as e1:
                    hf_error = e1
                    
                    # Method 2: Try chat_completion method (HuggingFace native)
                    try:
                        response = client.chat_completion(
                            messages=messages,
                            model=model,
                            temperature=temperature,
                            max_tokens=1000,
                        )
                        
                        # Handle different response formats
                        if hasattr(response, 'generated_text'):
                            return response.generated_text.strip()
                        elif isinstance(response, dict) and 'generated_text' in response:
                            return response['generated_text'].strip()
                        elif isinstance(response, list) and len(response) > 0:
                            if isinstance(response[0], dict) and 'generated_text' in response[0]:
                                return response[0]['generated_text'].strip()
                        
                        return str(response).strip()
                        
                    except Exception as e2:
                        # Both HuggingFace methods failed
                        hf_error = f"Method 1: {str(e1)}. Method 2: {str(e2)}"
                        raise APIError("HuggingFace", f"All HuggingFace methods failed. {hf_error}")
                
            except Exception as e:
                # HuggingFace failed, try fallback to Nebius
                if hf_error is None:
                    hf_error = str(e)
                logger.warning(f"HuggingFace API failed: {hf_error}, falling back to Nebius")
                
                try:
                    # Use Nebius model appropriate for the task
                    nebius_model = model_config.get_model_for_provider("question_enhancer", "nebius")
                    return make_nebius_completion(nebius_model, messages, temperature, response_format)
                except Exception as nebius_error:
                    raise APIError("HuggingFace", f"HuggingFace failed: {hf_error}. Nebius fallback also failed: {str(nebius_error)}")
        
        else:
            raise APIError("Config", f"Unsupported LLM provider: {provider}")
        
    except Exception as e:
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
            if response_format and response_format.get("type") == "json_object":
                kwargs["response_format"] = {"type": "json_object"}

            response = await client.chat.completions.create(**kwargs)

            if not response.choices:
                raise APIError("OpenAI", "No completion choices returned")

            content = response.choices[0].message.content
            if content is None:
                raise APIError("OpenAI", "Empty response content")

            return content.strip()

        elif provider == "anthropic":
            client = create_async_llm_client()
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
            # HuggingFace doesn't support async, fallback to Nebius
            logger.warning("HuggingFace does not support async operations, falling back to Nebius")
            
            try:
                # Use Nebius model appropriate for the task
                nebius_model = model_config.get_model_for_provider("question_enhancer", "nebius")
                return await make_async_nebius_completion(nebius_model, messages, temperature, response_format)
            except Exception as nebius_error:
                raise APIError("HuggingFace", f"HuggingFace async not supported. Nebius fallback failed: {str(nebius_error)}")

        else:
            raise APIError("Config", f"Unsupported LLM provider: {provider}")

    except Exception as e:
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
