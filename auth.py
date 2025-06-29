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
    Verify the provided API key against the stored key.
    
    Args:
        credentials: HTTP Bearer token from the request
        
    Returns:
        True if authentication is successful
        
    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    stored_api_key = get_api_key_from_env()
    if not stored_api_key:
        # In production, require API key. In development, allow without key.
        import os
        if os.getenv("ENVIRONMENT", "development") == "production":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required in production mode",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return True
    
    provided_key = credentials.credentials
    expected_hash = hash_api_key(stored_api_key)
    provided_hash = hash_api_key(provided_key)
    
    # Use constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(expected_hash, provided_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
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
    verify_api_key(credentials)
    
    stored_api_key = get_api_key_from_env()
    
    return {
        "authenticated": True,
        "mode": "development" if not stored_api_key else "production",
        "api_key_hash": hash_api_key(credentials.credentials)[:8] if credentials and credentials.credentials else "dev-mode"
    }