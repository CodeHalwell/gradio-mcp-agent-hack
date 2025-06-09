"""
Package management utilities for dynamic package installation in Modal sandboxes.
This module provides functions to analyze code for imports and manage package installation.
"""
import ast
import re
from typing import Set, List

try:
    from mcp_hub.logging_config import logger
except ImportError:
    # Fallback logger for testing/standalone use
    import logging
    logger = logging.getLogger(__name__)


# Core packages that should be preinstalled in the base image
CORE_PREINSTALLED_PACKAGES = {
    "numpy", "pandas", "matplotlib", "requests", "json", "os", "sys", 
    "time", "datetime", "math", "random", "collections", "itertools",
    "functools", "re", "urllib", "csv", "sqlite3", "pathlib", "typing",
    "asyncio", "threading", "multiprocessing", "subprocess", "shutil",
    "tempfile", "io", "gzip", "zipfile", "tarfile", "base64", "hashlib",
    "secrets", "uuid", "pickle", "copy", "operator", "bisect", "heapq",
    "contextlib", "weakref", "gc", "inspect", "types", "enum", "dataclasses",
    "decimal", "fractions", "statistics", "string", "textwrap", "locale",
    "calendar", "timeit", "argparse", "getopt", "logging", "warnings",
    "platform", "signal", "errno", "ctypes", "struct", "array", "queue",
    "socketserver", "http", "urllib2", "html", "xml", "email", "mailbox"
}

# Extended packages that can be dynamically installed
COMMON_PACKAGES = {
    "scikit-learn": "sklearn", 
    "beautifulsoup4": "bs4",
    "pillow": "PIL",
    "opencv-python-headless": "cv2",
    "python-dateutil": "dateutil",
    "plotly": "plotly",
    "seaborn": "seaborn",
    "polars": "polars",
    "lightgbm": "lightgbm", 
    "xgboost": "xgboost",
    "flask": "flask",
    "fastapi": "fastapi",
    "httpx": "httpx",
    "networkx": "networkx",
    "wordcloud": "wordcloud",
    "textblob": "textblob",
    "spacy": "spacy",
    "nltk": "nltk"
}

# Map import names to package names
IMPORT_TO_PACKAGE = {v: k for k, v in COMMON_PACKAGES.items()}
IMPORT_TO_PACKAGE.update({k: k for k in COMMON_PACKAGES.keys()})


def extract_imports_from_code(code_str: str) -> Set[str]:
    """
    Extract all import statements from Python code using AST parsing.
    
    Args:
        code_str: The Python code to analyze
        
    Returns:
        Set of imported module names (top-level only)
    """
    imports = set()
    
    try:
        tree = ast.parse(code_str)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Get top-level module name
                    module_name = alias.name.split('.')[0]
                    imports.add(module_name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Get top-level module name
                    module_name = node.module.split('.')[0]
                    imports.add(module_name)
    except Exception as e:
        logger.warning(f"Failed to parse code with AST, falling back to regex: {e}")
        # Fallback to regex-based extraction
        imports.update(extract_imports_with_regex(code_str))
    
    return imports


def extract_imports_with_regex(code_str: str) -> Set[str]:
    """
    Fallback method to extract imports using regex patterns.
    
    Args:
        code_str: The Python code to analyze
        
    Returns:
        Set of imported module names
    """
    imports = set()
    
    # Pattern for "import module" statements
    import_pattern = r'^import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)'
    
    # Pattern for "from module import ..." statements  
    from_pattern = r'^from\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import'
    
    for line in code_str.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Check for import statements
        import_match = re.match(import_pattern, line)
        if import_match:
            module_name = import_match.group(1).split('.')[0]
            imports.add(module_name)
            continue
            
        # Check for from...import statements
        from_match = re.match(from_pattern, line)
        if from_match:
            module_name = from_match.group(1).split('.')[0]
            imports.add(module_name)
            
    return imports


def get_packages_to_install(detected_imports: Set[str]) -> List[str]:
    """
    Determine which packages need to be installed based on detected imports.
    
    Args:
        detected_imports: Set of module names found in the code
        
    Returns:
        List of package names that need to be pip installed
    """
    packages_to_install = []
    
    for import_name in detected_imports:
        # Skip if it's a core preinstalled package
        if import_name in CORE_PREINSTALLED_PACKAGES:
            continue
            
        # Check if we have a known package mapping
        if import_name in IMPORT_TO_PACKAGE:
            package_name = IMPORT_TO_PACKAGE[import_name]
            packages_to_install.append(package_name)
        # For unknown packages, assume package name matches import name
        elif import_name not in CORE_PREINSTALLED_PACKAGES:
            packages_to_install.append(import_name)
    
    return packages_to_install


def get_warmup_import_commands() -> List[str]:
    """
    Get list of import commands to run during sandbox warmup.
    
    Returns:
        List of Python import statements for core packages
    """
    core_imports = [
        "import numpy",
        "import pandas", 
        "import matplotlib.pyplot",
        "import requests",
        "print('Core packages warmed up successfully')"
    ]
    
    return core_imports


def create_package_install_command(packages: List[str]) -> str:
    """
    Create a pip install command for the given packages.
    
    Args:
        packages: List of package names to install
        
    Returns:
        Pip install command string
    """
    if not packages:
        return ""
    
    # Remove duplicates and sort
    unique_packages = sorted(set(packages))
    return f"pip install {' '.join(unique_packages)}"