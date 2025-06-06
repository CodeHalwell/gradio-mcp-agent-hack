"""Utilfrom .exceptions import APIError, ValidationErrorty functions for the MCP Hub project."""

import json
import re
from typing import Dict, Any, List, Optional
from openai import OpenAI
from .config import api_config
from .exceptions import APIError, ValidationError

def create_nebius_client() -> OpenAI:
    """Create and return a Nebius OpenAI client."""
    return OpenAI(
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
