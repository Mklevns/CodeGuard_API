#!/bin/bash
# Startup script for CodeGuard API - handles both local and Cloud Run environments

# Set default port if not provided
if [ -z "$PORT" ]; then
    export PORT=8080
fi

# Install curl for health checks if not present
which curl > /dev/null || (apt-get update && apt-get install -y curl)

echo "Starting CodeGuard API on port $PORT"
echo "Environment: ${ENVIRONMENT:-production}"

# Start the application
exec python run.py