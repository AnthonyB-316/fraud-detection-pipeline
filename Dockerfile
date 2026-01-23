# Fraud Detection Pipeline
# Multi-stage build for API, Dashboard, and Kafka Consumer

FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY dashboard/ ./dashboard/

# Set Python path
ENV PYTHONPATH=/app/src:/app

# -------------------
# API Service
# -------------------
FROM base as api
COPY models/ ./models/
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# -------------------
# Dashboard Service
# -------------------
FROM base as dashboard
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

# -------------------
# Kafka Consumer Service
# -------------------
FROM base as kafka-consumer
COPY models/ ./models/
CMD ["python", "-m", "src.kafka_consumer"]

# -------------------
# Training Job
# -------------------
FROM base as training
COPY data/ ./data/
CMD ["python", "-m", "src.train"]
