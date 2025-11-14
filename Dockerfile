# Dockerfile for Cyber-Physical Systems MLflow Application
# Multi-stage build for optimized production image

# Build stage
FROM python:3.14-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY ml-models/requirements.txt ./ml-models/
COPY data-collection/requirements.txt ./data-collection/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r ml-models/requirements.txt && \
    pip install --no-cache-dir -r data-collection/requirements.txt

# Production stage
FROM python:3.14-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/mlruns /app/data /app/logs

# Set environment variables
ENV PYTHONPATH=/app:/app/ml-models:/app/data-collection
ENV MLFLOW_TRACKING_URI=file:///app/mlruns
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/app/mlruns

# Expose ports
EXPOSE 5000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command - can be overridden
CMD ["mlflow", "server", "--backend-store-uri", "file:///app/mlruns", "--default-artifact-root", "/app/mlruns", "--host", "0.0.0.0", "--port", "5000"]
