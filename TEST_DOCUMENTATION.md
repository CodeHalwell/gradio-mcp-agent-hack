# MCP Hub Test Suite Documentation

This document provides comprehensive instructions for running and understanding the **simplified** test suite for the MCP Hub project.

## Test Structure

The test suite has been streamlined to focus on core functionality with manageable, fast-running tests:

### Unit Tests (`tests/unit/`)
- `test_question_enhancer_agent.py` - QuestionEnhancerAgent core functionality tests
- `test_web_search_agent.py` - WebSearchAgent basic functionality tests  
- `test_llm_processor_agent.py` - LLMProcessorAgent content processing tests
- `test_citation_formatter_agent.py` - CitationFormatterAgent formatting tests
- `test_code_generator_agent.py` - CodeGeneratorAgent code generation tests

### Integration Tests (`tests/integration/`)
- Currently simplified - complex integration tests removed for manageability

## Key Features

### Simplified Mock-Based Approach
- **Fast execution** - All tests run in milliseconds without external dependencies
- **Reliable** - No network calls or complex service dependencies  
- **Easy to understand** - Clear mock implementations show expected behavior
- **Maintainable** - Simple test structure that's easy to extend

### Core Functionality Coverage
- Basic agent instantiation and method calls
- Success and error scenarios for each agent
- Input validation and edge cases
- Return value structure verification

## Running Tests

### Prerequisites

Ensure you have the testing dependencies installed:

```bash
pip install pytest pytest-cov pytest-asyncio pytest-mock
```

Or install from the project requirements:

```bash
pip install -r requirements.txt
```

## Running Tests

### Prerequisites

Ensure you have the testing dependencies installed:

```bash
pip install pytest pytest-cov pytest-asyncio pytest-mock
```

### Basic Test Execution

#### Run All Tests
```bash
pytest
```

#### Run Unit Tests Only
```bash
pytest tests/unit/
```

#### Run Specific Test File
```bash
pytest tests/unit/test_question_enhancer_agent.py
```

#### Run Specific Test Class
```bash
pytest tests/unit/test_question_enhancer_agent.py::TestQuestionEnhancerAgent
```

#### Run Specific Test Method
```bash
pytest tests/unit/test_question_enhancer_agent.py::TestQuestionEnhancerAgent::test_enhance_question_success
```

### Test Options and Configurations

#### Run with Verbose Output
```bash
pytest -v
```

#### Run without Coverage (Faster)
```bash
pytest --no-cov
```

#### Run Tests Excluding Slow Tests
```bash
pytest -m "not slow"
```

#### Run Only Slow Tests (Performance Tests)
```bash
pytest -m slow
```

#### Run Tests with Verbose Output
```bash
pytest -v
```

#### Run Tests and Stop on First Failure
```bash
pytest -x
```

#### Run Tests with Short Traceback
```bash
pytest --tb=short
```

#### Run Tests in Parallel (if pytest-xdist is installed)
```bash
pip install pytest-xdist
pytest -n auto
```

### Test Markers

The test suite uses pytest markers to categorize tests:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.async_test` - Async test cases
- `@pytest.mark.slow` - Slow running tests (performance tests)
- `@pytest.mark.requires_api` - Tests that need API keys (currently mocked)

## Test Coverage Goals

The test suite aims for:
- **80%+ overall code coverage** for `app.py` and `mcp_hub/` modules
- **100% coverage** for critical agent classes and orchestration logic
- **Comprehensive error handling** coverage for all failure scenarios
- **Both sync and async** code path coverage

### Checking Coverage

Generate a detailed coverage report:

```bash
pytest --cov=app --cov=mcp_hub --cov-report=html:htmlcov --cov-report=term-missing
```

Open the HTML report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Test Categories Explained

### Unit Tests

Unit tests focus on individual components in isolation:

- **Agent Classes**: Test each agent's core functionality with mocked dependencies
- **Utility Functions**: Test helper functions from `mcp_hub` modules
- **Configuration**: Test configuration loading and validation
- **Exception Handling**: Test custom exception classes

Key characteristics:
- Fast execution (< 1 second per test)
- Mocked external dependencies (APIs, Modal, etc.)
- Focused on single components
- High coverage of edge cases and error conditions

### Integration Tests

Integration tests focus on component interactions:

- **End-to-End Workflows**: Test complete orchestration flows
- **Async vs Sync Handling**: Test event loop management and fallbacks
- **Error Recovery**: Test system resilience under various failure conditions
- **UI Endpoints**: Test Gradio wrapper functions and MCP endpoints

Key characteristics:
- Moderate execution time (1-5 seconds per test)
- Test real component interactions
- Mocked only at system boundaries (external APIs)
- Focus on workflow correctness

### Performance Tests

Performance tests ensure the system can handle load:

- **Metrics Collection**: Test performance monitoring overhead
- **Concurrent Operations**: Test system behavior under concurrent load
- **Memory Management**: Test for memory leaks and resource cleanup
- **Stress Testing**: Test system limits and degradation

Key characteristics:
- Longer execution time (marked with `@pytest.mark.slow`)
- Test system behavior under load
- Focus on timing and resource usage
- May be skipped in CI for speed

## Environment Setup for Testing

The test suite uses environment variable mocking to avoid requiring real API keys:

### Automatic Mocking

The `conftest.py` file automatically sets up:
- `TAVILY_API_KEY=tvly-test-key-12345`
- `NEBIUS_API_KEY=test-nebius-key`
- `OPENAI_API_KEY=test-openai-key`
- `ANTHROPIC_API_KEY=test-anthropic-key`
- `HUGGINGFACE_API_KEY=test-hf-key`
- `LLM_PROVIDER=nebius`

### Manual Environment Setup

If you need to test with real APIs (not recommended for CI):

```bash
# Create .env file with real keys
cat > .env << EOF
TAVILY_API_KEY=your_real_tavily_key
NEBIUS_API_KEY=your_real_nebius_key
LLM_PROVIDER=nebius
EOF

# Run tests with real APIs (use caution!)
pytest tests/unit/test_web_search_agent.py -k "not mock"
```

## Debugging Tests

### Common Test Failures

1. **Import Errors**: 
   - Ensure all dependencies are installed
   - Check that `mcp_hub` package is importable

2. **Mock Issues**:
   - Verify mock setups match actual function signatures
   - Check that return values match expected formats

3. **Async Test Issues**:
   - Ensure `pytest-asyncio` is installed
   - Use `@pytest.mark.asyncio` for async tests

4. **Environment Issues**:
   - Clear pytest cache: `pytest --cache-clear`
   - Reset environment: `unset TAVILY_API_KEY; pytest`

### Debug Mode

Run tests with detailed output:

```bash
pytest -v -s --tb=long tests/unit/test_specific_test.py
```

Add debug prints in tests:

```python
def test_something():
    result = function_under_test()
    print(f"Debug: result = {result}")  # Will show with -s flag
    assert result["status"] == "success"
```

### Test Isolation

Ensure tests run independently:

```bash
# Run tests in random order to catch dependencies
pip install pytest-randomly
pytest --randomly-seed=auto
```

## Continuous Integration

The test suite is designed to run in CI environments:

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - uses: actions/setup-python@v2
      with:
        python-version: '3.12'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio pytest-mock
    - name: Run tests
      run: pytest --cov=app --cov=mcp_hub --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

### Docker Testing

```dockerfile
FROM python:3.12
COPY requirements.txt .
RUN pip install -r requirements.txt && pip install pytest pytest-cov pytest-asyncio pytest-mock
COPY . /app
WORKDIR /app
CMD ["pytest", "--cov=app", "--cov=mcp_hub"]
```

## Test Data and Fixtures

### Common Fixtures

The `conftest.py` provides reusable fixtures:

- `mock_tavily_client` - Mocked Tavily search client
- `mock_llm_response` - Standard LLM response format
- `mock_modal_sandbox` - Mocked Modal sandbox
- `sample_user_request` - Example user requests
- `sample_search_results` - Example search results
- `sample_code` - Example Python code

### Using Fixtures

```python
def test_with_fixtures(mock_tavily_client, sample_user_request):
    # Fixtures are automatically injected
    agent = WebSearchAgent()
    result = agent.search(sample_user_request)
    assert result["status"] == "success"
```

## Contributing to Tests

When adding new features:

1. **Write tests first** (TDD approach)
2. **Add unit tests** for new functions/classes
3. **Add integration tests** for new workflows
4. **Update this documentation** if needed
5. **Ensure coverage remains high**

### Test Naming Conventions

- `test_function_name_basic` - Basic functionality test
- `test_function_name_with_condition` - Test with specific condition
- `test_function_name_error_handling` - Error condition test
- `test_function_name_edge_case` - Edge case test

### Mock Guidelines

- Mock at the **system boundary** (external APIs, file system)
- **Don't over-mock** - test real object interactions when possible
- **Match real interfaces** - mocks should return realistic data
- **Use return values** that match actual API responses

## Troubleshooting

### Common Issues

1. **"Module not found" errors**:
   ```bash
   export PYTHONPATH=/path/to/gradio-mcp-agent-hack:$PYTHONPATH
   ```

2. **"Event loop is running" errors**:
   - Ensure proper async test setup
   - Use `asyncio_mode = "auto"` in pytest.ini

3. **Mock patching issues**:
   - Check import paths in patch decorators
   - Verify mock is applied before import

4. **Coverage not showing**:
   ```bash
   pip install coverage
   pytest --cov=. --cov-report=html
   ```

### Getting Help

- Check test output with `-v` flag for details
- Use `--tb=long` for full tracebacks
- Review mock setups in `conftest.py`
- Check actual vs expected return formats in failing tests

## Performance Considerations

### Test Speed Optimization

- Use `@pytest.mark.slow` for expensive tests
- Mock external API calls
- Use smaller test data sets
- Run fast tests first: `pytest tests/unit/ tests/integration/`

### Resource Cleanup

Tests automatically clean up:
- Mock objects and patches
- Temporary files (if any)
- Thread pools and async tasks

The test suite is designed to be:
- **Fast** - Most tests run in milliseconds
- **Reliable** - Consistent results across environments
- **Comprehensive** - High coverage of functionality
- **Maintainable** - Clear structure and documentation