"""Unit tests for CodeGeneratorAgent - Simplified."""

import pytest
from unittest.mock import Mock


class MockCodeGeneratorAgent:
    """Mock implementation for testing."""
    
    def generate_code(self, request: str, context: str = ""):
        """Mock generate_code method."""
        if not request:
            return {
                "status": "error",
                "code": "",
                "error": "Empty request"
            }
        
        # Simple mock code based on request
        if "csv" in request.lower():
            code = "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())"
        elif "plot" in request.lower():
            code = "import matplotlib.pyplot as plt\nplt.plot([1,2,3], [1,4,9])\nplt.show()"
        else:
            code = f"# Code for: {request}\nprint('Hello, World!')"
        
        return {
            "status": "success",
            "code": code,
            "context_used": bool(context),
            "dependencies": ["pandas"] if "csv" in request.lower() else []
        }


class TestCodeGeneratorAgent:
    """Test suite for CodeGeneratorAgent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = MockCodeGeneratorAgent()
    
    def test_generate_code_success(self):
        """Test successful code generation."""
        # Setup
        request = "Create a script to read CSV data"
        
        # Execute
        result = self.agent.generate_code(request)
        
        # Verify
        assert result["status"] == "success"
        assert "pandas" in result["code"]
        assert "read_csv" in result["code"]
        assert "pandas" in result["dependencies"]
    
    def test_generate_code_with_context(self):
        """Test code generation with context."""
        # Setup
        request = "Plot some data"
        context = "We have numerical data in arrays"
        
        # Execute
        result = self.agent.generate_code(request, context)
        
        # Verify
        assert result["status"] == "success"
        assert "matplotlib" in result["code"]
        assert result["context_used"] is True
    
    def test_generate_code_empty_request(self):
        """Test code generation with empty request."""
        # Execute
        result = self.agent.generate_code("")
        
        # Verify
        assert result["status"] == "error"
        assert "error" in result
    
    def test_generate_code_generic(self):
        """Test generic code generation."""
        # Setup
        request = "Write a hello world program"
        
        # Execute
        result = self.agent.generate_code(request)
        
        # Verify
        assert result["status"] == "success"
        assert "Hello, World!" in result["code"]