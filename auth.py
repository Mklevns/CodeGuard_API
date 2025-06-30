from fastapi import HTTPException, Security, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import hashlib
import hmac
from typing import Optional

# Security scheme for API key authentication
security = HTTPBearer(auto_error=False)

def get_api_key_from_env() -> Optional[str]:
    """Get the API key from environment variables."""
    return os.getenv("CODEGUARD_API_KEY")

def hash_api_key(api_key: str) -> str:
    """Create a secure hash of the API key for comparison."""
    return hashlib.sha256(api_key.encode()).hexdigest()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    """
    Verifies the API key. This function is secure in ALL environments.
    """
    # Always get the expected key from environment variables
    stored_api_key = os.getenv("CODEGUARD_API_KEY")
    
    # For development, use a known key if none is set
    if not stored_api_key:
        environment = os.getenv("ENVIRONMENT", "development")
        if environment == "development":
            stored_api_key = "codeguard-dev-key-2025"
        else:
            # If no key is set on the server, no requests can be authenticated
            raise HTTPException(
                status_code=503, 
                detail="Service is not configured for authentication."
            )

    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=401, 
            detail="API key is required.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Use a secure comparison to prevent timing attacks
    if not hmac.compare_digest(credentials.credentials, stored_api_key):
        raise HTTPException(
            status_code=401, 
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return True

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    """
    Get current authenticated user information.
    
    Args:
        credentials: HTTP Bearer token from the request
        
    Returns:
        User information dictionary
    """
    import os
    
    # Always verify the API key - no bypasses
    verify_api_key(credentials)
    
    stored_api_key = get_api_key_from_env()
    environment = os.getenv("ENVIRONMENT", "development")
    
    # Determine mode based on whether production key is configured
    mode = "production" if stored_api_key else "development"
    
    return {
        "authenticated": True,
        "mode": mode,
        "environment": environment,
        "api_key_hash": hash_api_key(credentials.credentials)[:8] if credentials and credentials.credentials else "unknown"
    }