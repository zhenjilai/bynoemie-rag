"""
Setup script for ByNoemie RAG Chatbot

Install in development mode:
    pip install -e .

Install with extras:
    pip install -e ".[dev]"
    pip install -e ".[server]"
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = [
        line.strip() 
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = []

# Core dependencies (minimal for basic functionality)
core_requirements = [
    "langchain>=0.3.0",
    "langchain-core>=0.3.0",
    "langgraph>=0.2.0",
    "langsmith>=0.1.0",
    "langchain-groq>=0.2.0",
    "groq>=0.11.0",
    "sentence-transformers>=3.0.0",
    "chromadb>=0.5.0",
    "pyyaml>=6.0.0",
    "pydantic>=2.0.0",
    "requests>=2.31.0",
    "python-dotenv>=1.0.0",
]

# Development dependencies
dev_requirements = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0.0",
    "black>=24.0.0",
    "ruff>=0.5.0",
    "mypy>=1.10.0",
    "ipython>=8.0.0",
    "jupyter>=1.0.0",
]

# Server dependencies
server_requirements = [
    "fastapi>=0.110.0",
    "uvicorn>=0.30.0",
    "sse-starlette>=2.0.0",
]

# All LLM providers
providers_requirements = [
    "langchain-openai>=0.2.0",
    "openai>=1.50.0",
    "langchain-anthropic>=0.2.0",
    "anthropic>=0.36.0",
    "langchain-ollama>=0.2.0",
]

setup(
    name="bynoemie-rag-chatbot",
    version="1.0.0",
    author="ByNoemie",
    author_email="dev@bynoemie.com",
    description="A production-ready RAG chatbot with LangChain, LangGraph, and LangSmith integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bynoemie/rag-chatbot",
    
    # Package discovery
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "examples"]),
    package_dir={"": "."},
    
    # Include data files
    include_package_data=True,
    package_data={
        "config": ["*.yaml"],
        "data": ["prompts/*.txt", "prompts/*.yaml"],
    },
    
    # Dependencies
    python_requires=">=3.10",
    install_requires=core_requirements,
    
    # Optional dependencies
    extras_require={
        "dev": dev_requirements,
        "server": server_requirements,
        "providers": providers_requirements,
        "all": dev_requirements + server_requirements + providers_requirements,
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "bynoemie-chat=main:main",
            "bynoemie-process=main:process_data",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Keywords
    keywords=[
        "rag",
        "chatbot",
        "langchain",
        "langgraph",
        "langsmith",
        "llm",
        "ai",
        "fashion",
        "ecommerce",
    ],
)
