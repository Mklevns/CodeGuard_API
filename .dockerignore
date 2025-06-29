FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic flake8

# Copy application code
COPY . .

# Make startup script executable
RUN chmod +x start.sh

# Expose port (Cloud Run uses PORT environment variable)
EXPOSE 8080

# Health check for Cloud Run
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Run the application using startup script
CMD ["./start.sh"]