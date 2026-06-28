# ==============================================
# x-llamaindex Dockerfile
# ==============================================
# Multi-stage build for optimized image size

# Stage 1: Builder
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --system -e .

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

# Labels
LABEL maintainer="John Young <john.young@foxmail.com>"
LABEL description="x-llamaindex - Enterprise RAG System"
LABEL version="0.1.0"

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser data ./data
COPY --chown=appuser:appuser config.yaml.example ./config.yaml.example

# Create necessary directories
RUN mkdir -p logs storage/chroma && \
    chown -R appuser:appuser logs storage

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# Run the application
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
