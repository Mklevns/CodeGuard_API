#!/usr/bin/env python3
"""
Production entry point for CodeGuard API.
Optimized for reliable deployment with proper error handling.
"""

import os
import sys
import uvicorn
from pathlib import Path

def main():
    """Main entry point for production deployment."""
    # Ensure we're in the correct directory
    app_dir = Path(__file__).parent
    os.chdir(app_dir)
    
    # Add current directory to Python path
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    
    # Set production environment variables
    os.environ.setdefault("ENVIRONMENT", "production")
    
    # Configure port - try PORT env var, fallback to 8080
    port = int(os.environ.get("PORT", 8080))
    
    print(f"Starting CodeGuard API on 0.0.0.0:{port} (production mode)")
    
    try:
        # Import the FastAPI app
        from main import app
        
        # Run with production settings
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            reload=False,
            access_log=True,
            log_level="info"
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()