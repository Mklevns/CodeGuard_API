#!/usr/bin/env python3
"""
Production entry point for CodeGuard API.
This script handles Cloud Run deployment requirements.
"""

import os
import uvicorn
from main import app

if __name__ == "__main__":
    # Cloud Run sets PORT environment variable
    port = int(os.environ.get("PORT", 8080))
    host = "0.0.0.0"
    
    print(f"Starting CodeGuard API on {host}:{port}")
    
    # Production configuration for Cloud Run
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        reload=False  # No reload in production
    )