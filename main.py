from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import os
from models import AuditRequest, AuditResponse
from audit import analyze_code
from auth import verify_api_key, get_current_user
from rule_loader import CustomRuleEngine

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
    """Root endpoint providing basic service information and health check."""
    return {
        "status": "healthy",
        "service": "CodeGuard API",
        "version": "1.0.0",
        "description": "Static code analysis service for ML/RL Python code",
        "endpoints": {
            "audit": "/audit",
            "openapi": "/.well-known/openapi.yaml",
            "deployment_openapi": "/openapi-deployment.yaml",
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

@app.get("/openapi-deployment.yaml")
async def get_deployment_openapi_spec():
    """Serve the deployment OpenAPI specification for production use."""
    openapi_path = "openapi-deployment.yaml"
    if os.path.exists(openapi_path):
        return FileResponse(openapi_path, media_type="application/yaml")
    else:
        raise HTTPException(status_code=404, detail="Deployment OpenAPI specification not found")

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

@app.get("/rules/summary")
async def get_rules_summary():
    """Get a summary of all loaded custom rules."""
    try:
        engine = CustomRuleEngine()
        return {
            "status": "success",
            "summary": engine.get_rule_summary()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get rules summary: {str(e)}")

@app.post("/rules/reload")
async def reload_rules():
    """Reload all custom rules from rule files."""
    try:
        engine = CustomRuleEngine()
        engine.reload_rules()
        return {
            "status": "success",
            "message": f"Reloaded {engine.get_rule_count()} rules successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload rules: {str(e)}")

@app.get("/rules/by-tag/{tag}")
async def get_rules_by_tag(tag: str):
    """Get all rules that contain a specific tag."""
    try:
        engine = CustomRuleEngine()
        rules = engine.get_rules_by_tag(tag)
        return {
            "status": "success",
            "tag": tag,
            "count": len(rules),
            "rules": rules
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get rules by tag: {str(e)}")

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
    # Run the application - optimized for both development and Cloud Run
    environment = os.environ.get("ENVIRONMENT", "development")
    
    # Set appropriate default port based on environment
    if environment == "production":
        default_port = 8080  # Cloud Run standard port
    else:
        default_port = 5000  # Replit development port
    
    port = int(os.environ.get("PORT", default_port))
    
    print(f"Starting CodeGuard API on 0.0.0.0:{port} (environment: {environment})")
    
    if environment == "development":
        # Use import string for reload functionality in development
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            reload=True,
            log_level="info",
            access_log=True
        )
    else:
        # Use app object directly in production for better performance
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            reload=False,
            log_level="info",
            access_log=True
        )
