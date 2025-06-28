from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import os
from models import AuditRequest, AuditResponse
from audit import analyze_code
from auth import verify_api_key, get_current_user

# Create FastAPI app
app = FastAPI(
    title="CodeGuard API",
    version="1.0.0",
    description="Audits ML and RL Python files for issues using static analysis tools.",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint providing basic service information."""
    return {
        "service": "CodeGuard API",
        "version": "1.0.0",
        "description": "Static code analysis service for ML/RL Python code",
        "endpoints": {
            "audit": "/audit",
            "openapi": "/.well-known/openapi.yaml",
            "docs": "/docs",
            "privacy": "/privacy-policy",
            "terms": "/terms-of-service"
        }
    }

@app.post("/audit", response_model=AuditResponse)
async def audit_code(request: AuditRequest, current_user: dict = Depends(get_current_user)):
    """
    Analyzes submitted Python code files and returns a report of issues and suggestions.
    
    Args:
        request: AuditRequest containing files to analyze
        current_user: Authenticated user information
        
    Returns:
        AuditResponse with summary, issues, and fix suggestions
        
    Raises:
        HTTPException: If analysis fails
    """
    try:
        return analyze_code(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/.well-known/openapi.yaml")
async def get_openapi_spec():
    """Serve the OpenAPI specification for GPT Action integration."""
    openapi_path = ".well-known/openapi.yaml"
    if os.path.exists(openapi_path):
        return FileResponse(openapi_path, media_type="application/yaml")
    else:
        raise HTTPException(status_code=404, detail="OpenAPI specification not found")

@app.get("/auth/status")
async def auth_status(current_user: dict = Depends(get_current_user)):
    """Check authentication status and return user information."""
    return {
        "authenticated": True,
        "message": "API key is valid",
        "user_info": current_user
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "CodeGuard API"}

@app.get("/privacy-policy")
async def privacy_policy():
    """Serve the privacy policy for the CodeGuard API service."""
    privacy_path = "privacy-policy.md"
    if os.path.exists(privacy_path):
        return FileResponse(privacy_path, media_type="text/markdown")
    else:
        raise HTTPException(status_code=404, detail="Privacy policy not found")

@app.get("/terms-of-service")
async def terms_of_service():
    """Serve the terms of service for the CodeGuard API service."""
    terms_path = "terms-of-service.md"
    if os.path.exists(terms_path):
        return FileResponse(terms_path, media_type="text/markdown")
    else:
        raise HTTPException(status_code=404, detail="Terms of service not found")

if __name__ == "__main__":
    # Run the application
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
