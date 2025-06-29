#!/bin/bash

# CodeGuard API startup script
# This script ensures the server starts correctly in all environments

echo "Starting CodeGuard API..."

# Set default port if not specified
export PORT=${PORT:-5000}

# Start the Python application
exec python main.py