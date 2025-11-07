# MCP Hub API Documentation

Complete API documentation for the MCP Hub multi-agent code assistance system.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Authentication](#authentication)
- [Agent Endpoints](#agent-endpoints)
- [Monitoring Endpoints](#monitoring-endpoints)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Code Examples](#code-examples)
- [OpenAPI Specification](#openapi-specification)
- [MCP Protocol Schema](#mcp-protocol-schema)

## Overview

The MCP Hub provides a comprehensive REST API for multi-agent code assistance. The system coordinates 7 specialized agents to provide end-to-end support for research, code generation, and execution.

### Base URLs

- **Local Development**: `http://localhost:7860`
- **Production**: Configure based on your deployment

### API Format

All API endpoints follow Gradio's API format:
- **Method**: POST
- **Content-Type**: `application/json`
- **Request Body**: `{ "data": [...] }`
- **Response Body**: `{ "data": [...] }`

## Getting Started

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```bash
# Required
TAVILY_API_KEY=tvly-your-key-here

# LLM Provider (choose one)
LLM_PROVIDER=nebius  # or openai, anthropic, huggingface
NEBIUS_API_KEY=your-nebius-key
# OPENAI_API_KEY=your-openai-key
# ANTHROPIC_API_KEY=your-anthropic-key
# HUGGINGFACE_API_KEY=your-hf-key

# Optional: Caching
CACHE_BACKEND=file  # or redis
# REDIS_URL=redis://localhost:6379/0
```

3. Start the server:
```bash
python app.py
```

### Quick Start Example

```python
import requests

# Make a request to the question enhancer
response = requests.post(
    'http://localhost:7860/api/agent_question_enhancer_service',
    json={
        "data": ["How do I use async/await in Python?", 3]
    }
)

result = response.json()
print(result['data'][0]['sub_questions'])
```

## Authentication

Currently, the API uses API keys configured via environment variables. Include your provider API keys in the `.env` file.

### Future Authentication

API key authentication via headers may be added in future versions:
```bash
X-API-Key: your-api-key-here
```

## Agent Endpoints

### 1. Question Enhancer Agent

**Endpoint**: `/api/agent_question_enhancer_service`

**Description**: Breaks down complex questions into focused sub-questions for research.

**Request**:
```json
{
  "data": [
    "How do I implement async/await in Python?",
    3
  ]
}
```

**Parameters**:
- `data[0]` (string, required): User question (5-10000 chars)
- `data[1]` (integer, optional): Number of sub-questions (default: 3, max: 10)

**Response**:
```json
{
  "data": [
    {
      "sub_questions": [
        "What is the async/await syntax in Python?",
        "How do I create and run async functions?",
        "What are common use cases for asyncio?"
      ],
      "original_question": "How do I implement async/await in Python?"
    }
  ]
}
```

**Example**:
```python
response = requests.post(
    'http://localhost:7860/api/agent_question_enhancer_service',
    json={"data": ["Python decorators explained", 5]}
)
```

---

### 2. Web Search Agent

**Endpoint**: `/api/agent_web_search_service`

**Description**: Performs web searches using Tavily API.

**Request**:
```json
{
  "data": ["Python async programming tutorial"]
}
```

**Parameters**:
- `data[0]` (string, required): Search query (1-1000 chars)

**Response**:
```json
{
  "data": [
    {
      "results": [
        {
          "title": "Async Programming in Python",
          "url": "https://example.com/async-tutorial",
          "content": "Learn about async/await...",
          "score": 0.95
        }
      ],
      "query": "Python async programming tutorial",
      "answer": "Async programming in Python uses asyncio..."
    }
  ]
}
```

**Features**:
- Cached results (600s TTL)
- Rate limiting
- Circuit breaker protection

**Example**:
```python
response = requests.post(
    'http://localhost:7860/api/agent_web_search_service',
    json={"data": ["machine learning python scikit-learn"]}
)
```

---

### 3. LLM Processor Agent

**Endpoint**: `/api/agent_llm_processor_service`

**Description**: Processes text using LLM for various tasks.

**Request**:
```json
{
  "data": [
    "Long article content here...",
    "summarize",
    "Python async programming"
  ]
}
```

**Parameters**:
- `data[0]` (string, required): Text input (1-50000 chars)
- `data[1]` (string, required): Task type (`summarize`, `reason`, `extract_keywords`)
- `data[2]` (string, optional): Additional context

**Response**:
```json
{
  "data": [
    {
      "result": "This article explains async programming...",
      "task": "summarize"
    }
  ]
}
```

**Tasks**:

1. **Summarize**: Condense text to key points
   ```python
   response = requests.post(url, json={
       "data": [long_text, "summarize", "focus on main concepts"]
   })
   ```

2. **Reason**: Draw conclusions from facts
   ```python
   response = requests.post(url, json={
       "data": [facts, "reason", "analyze implications"]
   })
   ```

3. **Extract Keywords**: Extract important terms
   ```python
   response = requests.post(url, json={
       "data": [document, "extract_keywords", ""]
   })
   ```

---

### 4. Citation Formatter Agent

**Endpoint**: `/api/agent_citation_formatter_service`

**Description**: Formats URLs into APA-style citations.

**Request**:
```json
{
  "data": ["https://example.com/article"]
}
```

**Parameters**:
- `data[0]` (string, required): URL to format

**Response**:
```json
{
  "data": [
    "Example.com. (2025). Article Title. Retrieved from https://example.com/article"
  ]
}
```

**Example**:
```python
response = requests.post(
    'http://localhost:7860/api/agent_citation_formatter_service',
    json={"data": ["https://python.org/doc"]}
)
```

---

### 5. Code Generator Agent

**Endpoint**: `/api/agent_code_generator_service`

**Description**: Generates secure Python code with AST validation.

**Request**:
```json
{
  "data": [
    "Create a function to calculate fibonacci numbers",
    "The function should be recursive and efficient"
  ]
}
```

**Parameters**:
- `data[0]` (string, required): Code generation request (5-10000 chars)
- `data[1]` (string, required): Grounded context (1-50000 chars)

**Response**:
```json
{
  "data": [
    {
      "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
      "validation_passed": true,
      "imports": [],
      "disallowed_functions": []
    }
  ]
}
```

**Security Features**:
- AST validation
- Disallowed functions check (eval, exec, etc.)
- Input validation with XSS detection

**Disallowed Functions**:
- `eval`, `exec`, `compile`
- `open`, `__import__`
- `globals`, `locals`
- `getattr`, `setattr`, `delattr`

**Example**:
```python
response = requests.post(
    'http://localhost:7860/api/agent_code_generator_service',
    json={
        "data": [
            "Create a function to sort a list using quicksort",
            "Implementation should be in-place and recursive"
        ]
    }
)

if response.json()['data'][0]['validation_passed']:
    code = response.json()['data'][0]['code']
    print(code)
```

---

### 6. Code Runner Agent

**Endpoint**: `/api/agent_code_runner_service`

**Description**: Executes Python code in secure Modal sandboxes.

**Request**:
```json
{
  "data": [
    "print('Hello, World!')\nfor i in range(5):\n    print(i)"
  ]
}
```

**Parameters**:
- `data[0]` (string, required): Python code to execute (1-50000 chars)

**Response**:
```json
{
  "data": [
    "Hello, World!\n0\n1\n2\n3\n4\n"
  ]
}
```

**Safety Features**:
- Sandboxed execution (Modal)
- 60-second timeout
- Safety shim wrapping
- Output capture (stdout + stderr)

**Limitations**:
- No file system access
- No network access (unless explicitly enabled)
- Memory limits
- CPU limits

**Example**:
```python
code = """
import math

def calculate_area(radius):
    return math.pi * radius ** 2

print(f"Area of circle with radius 5: {calculate_area(5)}")
"""

response = requests.post(
    'http://localhost:7860/api/agent_code_runner_service',
    json={"data": [code]}
)

print(response.json()['data'][0])
```

---

## Monitoring Endpoints

### 1. Health Status

**Endpoint**: `/api/get_health_status_service`

**Description**: Comprehensive system health check.

**Request**:
```json
{
  "data": []
}
```

**Response**:
```json
{
  "data": [
    {
      "status": "healthy",
      "timestamp": "2025-01-06T10:30:00Z",
      "uptime_seconds": 3600,
      "system": {
        "cpu_percent": 45.5,
        "memory_percent": 60.0,
        "memory_available_mb": 4096,
        "disk_percent": 70.0,
        "disk_free_gb": 100
      },
      "api_connectivity": {
        "nebius": {
          "status": "healthy",
          "response_time_ms": 150
        },
        "tavily": {
          "status": "healthy",
          "response_time_ms": 200
        }
      }
    }
  ]
}
```

**Status Values**:
- `healthy`: All systems operational
- `degraded`: Some warnings (high CPU/memory)
- `unhealthy`: Critical issues detected

---

### 2. Performance Metrics

**Endpoint**: `/api/get_performance_metrics_service`

**Description**: Performance metrics and analytics.

**Request**:
```json
{
  "data": []
}
```

**Response**:
```json
{
  "data": [
    {
      "web_search_duration_seconds": {
        "count": 150,
        "average": 1.5,
        "min": 0.5,
        "max": 3.0,
        "latest": 1.2
      },
      "code_generation_duration_seconds": {
        "count": 50,
        "average": 5.2,
        "min": 2.1,
        "max": 10.5,
        "latest": 4.8
      }
    }
  ]
}
```

---

### 3. Cache Status

**Endpoint**: `/api/get_cache_status_service`

**Description**: Cache statistics and status.

**Request**:
```json
{
  "data": []
}
```

**Response** (File Backend):
```json
{
  "data": [
    {
      "status": "healthy",
      "backend": "file",
      "cache_dir": "cache",
      "total_files": 150,
      "expired_files": 5,
      "total_size_mb": 45.2,
      "default_ttl_seconds": 3600
    }
  ]
}
```

**Response** (Redis Backend):
```json
{
  "data": [
    {
      "status": "healthy",
      "backend": "redis",
      "redis_version": "7.0.0",
      "connected_clients": 10,
      "total_keys": 250,
      "memory_used_human": "15M",
      "default_ttl_seconds": 3600
    }
  ]
}
```

---

### 4. Sandbox Pool Status

**Endpoint**: `/api/get_sandbox_pool_status_service`

**Description**: Modal sandbox pool statistics.

**Request**:
```json
{
  "data": []
}
```

**Response**:
```json
{
  "data": [
    {
      "warm_sandboxes": 3,
      "total_executions": 127,
      "average_execution_time": 2.5,
      "pool_size": 5
    }
  ]
}
```

---

### 5. Prometheus Metrics

**Endpoint**: `/api/get_prometheus_metrics_service`

**Description**: Metrics in Prometheus text format.

**Request**:
```json
{
  "data": []
}
```

**Response** (text/plain):
```
# HELP mcp_hub_requests_total Total number of requests
# TYPE mcp_hub_requests_total counter
mcp_hub_requests_total{agent="web_search",operation="search"} 150

# HELP mcp_hub_cpu_usage_percent CPU usage percentage
# TYPE mcp_hub_cpu_usage_percent gauge
mcp_hub_cpu_usage_percent 45.5

# HELP mcp_hub_request_duration_seconds Request duration
# TYPE mcp_hub_request_duration_seconds histogram
mcp_hub_request_duration_seconds_bucket{agent="web_search",operation="search",le="0.5"} 50
mcp_hub_request_duration_seconds_bucket{agent="web_search",operation="search",le="1.0"} 120
mcp_hub_request_duration_seconds_bucket{agent="web_search",operation="search",le="+Inf"} 150
```

---

## Error Handling

### Error Response Format

All errors return a consistent format:

```json
{
  "data": [
    {
      "error": "Sanitized error message",
      "status": "error",
      "timestamp": "2025-01-06T10:30:00Z"
    }
  ]
}
```

### Common Error Codes

- **400 Bad Request**: Invalid input, validation failed
- **500 Internal Server Error**: Server-side error, API failure

### Error Messages

Error messages are sanitized to prevent information leakage:
- File paths removed
- API keys redacted
- IP addresses masked

### Example Error Handling

```python
try:
    response = requests.post(url, json={"data": [invalid_input]})
    response.raise_for_status()
    result = response.json()

    if 'error' in result.get('data', [{}])[0]:
        print(f"Error: {result['data'][0]['error']}")
    else:
        # Process successful result
        pass

except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
```

---

## Rate Limiting

### API Rate Limits

Rate limits depend on the underlying service providers:

**LLM Providers**:
- Nebius: Provider-specific limits
- OpenAI: Based on your plan
- Anthropic: Based on your plan
- HuggingFace: Free tier limitations

**Tavily Search**:
- Based on your Tavily plan
- Typically 100-1000 requests/month

### Retry Logic

The system includes automatic retry with exponential backoff:

**LLM API Calls**:
- Max attempts: 3
- Base delay: 2 seconds
- Max delay: 30 seconds

**Search API Calls**:
- Max attempts: 3
- Base delay: 1 second
- Max delay: 10 seconds

### Rate Limit Headers

Currently not implemented, but may be added in future versions.

---

## Code Examples

### Python

#### Basic Agent Usage

```python
import requests

BASE_URL = "http://localhost:7860"

class MCPHubClient:
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url

    def enhance_question(self, question, num_questions=3):
        """Enhance a question into sub-questions."""
        response = requests.post(
            f"{self.base_url}/api/agent_question_enhancer_service",
            json={"data": [question, num_questions]}
        )
        return response.json()['data'][0]

    def search_web(self, query):
        """Perform web search."""
        response = requests.post(
            f"{self.base_url}/api/agent_web_search_service",
            json={"data": [query]}
        )
        return response.json()['data'][0]

    def generate_code(self, request, context):
        """Generate Python code."""
        response = requests.post(
            f"{self.base_url}/api/agent_code_generator_service",
            json={"data": [request, context]}
        )
        return response.json()['data'][0]

    def run_code(self, code):
        """Execute Python code."""
        response = requests.post(
            f"{self.base_url}/api/agent_code_runner_service",
            json={"data": [code]}
        )
        return response.json()['data'][0]

# Usage
client = MCPHubClient()

# Enhance question
result = client.enhance_question("How do I use pandas for data analysis?")
print("Sub-questions:", result['sub_questions'])

# Search web
search_results = client.search_web("pandas dataframe tutorial")
print("Found:", len(search_results['results']), "results")

# Generate and run code
code_result = client.generate_code(
    "Create a function to merge two sorted lists",
    "Should be O(n+m) time complexity"
)

if code_result['validation_passed']:
    output = client.run_code(code_result['code'])
    print("Output:", output)
```

#### Async Usage

```python
import aiohttp
import asyncio

async def async_search(query):
    """Async web search."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:7860/api/agent_web_search_service",
            json={"data": [query]}
        ) as response:
            result = await response.json()
            return result['data'][0]

# Run multiple searches concurrently
async def main():
    queries = [
        "Python async programming",
        "Django REST framework",
        "Docker compose tutorial"
    ]

    tasks = [async_search(q) for q in queries]
    results = await asyncio.gather(*tasks)

    for query, result in zip(queries, results):
        print(f"{query}: {len(result['results'])} results")

asyncio.run(main())
```

### JavaScript/TypeScript

```javascript
// Using fetch API
class MCPHubClient {
  constructor(baseUrl = 'http://localhost:7860') {
    this.baseUrl = baseUrl;
  }

  async enhanceQuestion(question, numQuestions = 3) {
    const response = await fetch(
      `${this.baseUrl}/api/agent_question_enhancer_service`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: [question, numQuestions] })
      }
    );
    const result = await response.json();
    return result.data[0];
  }

  async searchWeb(query) {
    const response = await fetch(
      `${this.baseUrl}/api/agent_web_search_service`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: [query] })
      }
    );
    const result = await response.json();
    return result.data[0];
  }

  async generateCode(request, context) {
    const response = await fetch(
      `${this.baseUrl}/api/agent_code_generator_service`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: [request, context] })
      }
    );
    const result = await response.json();
    return result.data[0];
  }
}

// Usage
const client = new MCPHubClient();

(async () => {
  // Enhance question
  const enhanced = await client.enhanceQuestion(
    'How do I use React hooks?'
  );
  console.log('Sub-questions:', enhanced.sub_questions);

  // Search web
  const searchResult = await client.searchWeb('React hooks tutorial');
  console.log('Found:', searchResult.results.length, 'results');
})();
```

### curl

```bash
# Enhance question
curl -X POST http://localhost:7860/api/agent_question_enhancer_service \
  -H "Content-Type: application/json" \
  -d '{"data": ["How do I use Docker?", 3]}'

# Search web
curl -X POST http://localhost:7860/api/agent_web_search_service \
  -H "Content-Type: application/json" \
  -d '{"data": ["Docker tutorial"]}'

# Generate code
curl -X POST http://localhost:7860/api/agent_code_generator_service \
  -H "Content-Type: application/json" \
  -d '{"data": ["Create a function to reverse a string", "Should work with unicode"]}'

# Run code
curl -X POST http://localhost:7860/api/agent_code_runner_service \
  -H "Content-Type: application/json" \
  -d '{"data": ["print(\"Hello, World!\")"]}'

# Get health status
curl -X POST http://localhost:7860/api/get_health_status_service \
  -H "Content-Type: application/json" \
  -d '{"data": []}'

# Get Prometheus metrics
curl -X POST http://localhost:7860/api/get_prometheus_metrics_service \
  -H "Content-Type: application/json" \
  -d '{"data": []}'
```

---

## OpenAPI Specification

Full OpenAPI 3.1 specification is available in `docs/openapi.yaml`.

### Importing into Tools

**Postman**:
1. Open Postman
2. Import → Upload Files
3. Select `docs/openapi.yaml`

**Swagger UI**:
```bash
docker run -p 8080:8080 -e SWAGGER_JSON=/docs/openapi.yaml \
  -v $(pwd)/docs:/docs swaggerapi/swagger-ui
```

**Insomnia**:
1. Create New Document
2. Import from File
3. Select `docs/openapi.yaml`

---

## MCP Protocol Schema

Full MCP protocol schema is available in `docs/mcp_schema.json`.

### Schema Validation

Validate requests/responses against the schema:

```python
import json
import jsonschema

# Load schema
with open('docs/mcp_schema.json') as f:
    schema = json.load(f)

# Validate request
request = {
    "user_request": "How do I use async/await?",
    "num_questions": 3
}

agent_schema = next(
    a for a in schema['capabilities']['agents']
    if a['name'] == 'question_enhancer'
)

jsonschema.validate(request, agent_schema['input_schema'])
```

---

## Best Practices

### 1. Error Handling

Always handle errors gracefully:

```python
try:
    response = client.generate_code(request, context)
    if response.get('validation_passed'):
        # Use the code
        pass
    else:
        # Handle validation failure
        print("Code validation failed")
except Exception as e:
    # Handle API error
    print(f"Error: {e}")
```

### 2. Input Validation

Validate inputs before sending:

```python
def validate_code_request(request, context):
    if not request or len(request) < 5:
        raise ValueError("Request too short")
    if len(request) > 10000:
        raise ValueError("Request too long")
    # Additional validation...
```

### 3. Timeout Handling

Set appropriate timeouts:

```python
response = requests.post(
    url,
    json={"data": [query]},
    timeout=30  # 30 second timeout
)
```

### 4. Retry Logic

Implement retries for transient failures:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def call_api(url, data):
    response = requests.post(url, json={"data": data})
    response.raise_for_status()
    return response.json()
```

### 5. Caching

Cache results when appropriate:

```python
import functools

@functools.lru_cache(maxsize=128)
def search_cached(query):
    return client.search_web(query)
```

---

## Troubleshooting

### Connection Refused

**Problem**: Cannot connect to `http://localhost:7860`

**Solution**:
- Verify server is running: `ps aux | grep python`
- Check if port is in use: `lsof -i :7860`
- Try alternative port: `python app.py --server-port 7861`

### API Key Errors

**Problem**: `TAVILY_API_KEY is required`

**Solution**:
- Check `.env` file exists
- Verify key format: `tvly-...`
- Reload environment: `source .env`

### Timeout Errors

**Problem**: Requests timing out

**Solution**:
- Increase timeout: `timeout=60`
- Check network connectivity
- Verify provider API status
- Monitor system resources

### Rate Limit Exceeded

**Problem**: Too many requests to external APIs

**Solution**:
- Use caching to reduce API calls
- Implement request throttling
- Upgrade provider plan
- Use Redis cache for distributed deployments

---

## Support and Resources

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Full docs in `docs/` directory
- **OpenAPI Spec**: `docs/openapi.yaml`
- **MCP Schema**: `docs/mcp_schema.json`
- **Examples**: `examples/` directory (if available)

---

## Changelog

### Version 1.0.0 (2025-01-06)

- Initial API documentation
- OpenAPI 3.1 specification
- MCP protocol schema
- Comprehensive examples
- Error handling guidelines
