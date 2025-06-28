#!/usr/bin/env python3
"""
Production optimized entry point for CodeGuard API.
Simplified version for reliable Cloud Run deployment.
"""

import os
import uvicorn
from main import app

if __name__ == "__main__":
    # Get port from environment (Cloud Run sets this automatically)
    port = int(os.environ.get("PORT", 8080))
    
    # Set production environment
    os.environ["ENVIRONMENT"] = "production"
    
    print(f"CodeGuard API starting on 0.0.0.0:{port} (production mode)")
    
    # Start with minimal production configuration
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False
    )