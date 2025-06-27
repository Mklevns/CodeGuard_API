from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import os
from models import AuditRequest, AuditResponse
from audit import analyze_code

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
            "docs": "/docs"
        }
    }

@app.post("/audit", response_model=AuditResponse)
async def audit_code(request: AuditRequest):
    """
    Analyzes submitted Python code files and returns a report of issues and suggestions.
    
    Args:
        request: AuditRequest containing files to analyze
        
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

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "CodeGuard API"}

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
