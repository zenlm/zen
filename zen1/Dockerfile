# Zen-1 Fine-tuned Model Docker Image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for models and data
RUN mkdir -p /app/checkpoints /app/data /app/models

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV MLX_CACHE_DIR=/app/models
ENV HF_HOME=/app/models

# Default command - inference server
CMD ["python", "-m", "uvicorn", "mlx_server:app", "--host", "0.0.0.0", "--port", "8000"]