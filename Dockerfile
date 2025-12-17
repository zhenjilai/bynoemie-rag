# ByNoemie RAG Chatbot - Docker Configuration
# Multi-stage build for smaller final image

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY config/ ./config/
COPY src/ ./src/
COPY examples/ ./examples/
COPY data/ ./data/
COPY main.py ./
COPY setup.py ./
COPY README.md ./

# Create data directories
RUN mkdir -p data/cache data/outputs data/embeddings data/products data/stock \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "main.py", "--mode", "serve"]

# =============================================================================
# Alternative: Development Stage
# =============================================================================
FROM runtime as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    ruff \
    ipython \
    jupyter

# Switch back to appuser
USER appuser

# Development command
CMD ["python", "-m", "ipython"]

# =============================================================================
# Usage:
# =============================================================================
# Build production image:
#   docker build -t bynoemie-rag .
#
# Build development image:
#   docker build --target development -t bynoemie-rag:dev .
#
# Run with API key:
#   docker run -e GROQ_API_KEY=$GROQ_API_KEY -p 8000:8000 bynoemie-rag
#
# Run interactive:
#   docker run -it -e GROQ_API_KEY=$GROQ_API_KEY bynoemie-rag:dev
#
# Run with volume mount for development:
#   docker run -it -v $(pwd):/app -e GROQ_API_KEY=$GROQ_API_KEY bynoemie-rag:dev
# =============================================================================
