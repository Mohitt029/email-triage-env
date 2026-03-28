FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY server/ server/
COPY *.py .
COPY openenv.yaml .

# Create non-root user for security
RUN useradd -m -u 1000 openenv && chown -R openenv:openenv /app
USER openenv

# Environment variables
ENV HOST=0.0.0.0
ENV PORT=7860
ENV WORKERS=2

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the server
CMD uvicorn server.app:app --host $HOST --port $PORT --workers $WORKERS