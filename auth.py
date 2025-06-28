from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import hashlib
import hmac
from typing import Optional

# Security scheme for API key authentication
security = HTTPBearer()

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
            detail="Missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    stored_api_key = get_api_key_from_env()
    
    if not stored_api_key:
        # If no API key is set in environment, allow access (development mode)
        return True
    
    provided_token = credentials.credentials
    
    # Use secure comparison to prevent timing attacks
    if not hmac.compare_digest(provided_token, stored_api_key):
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
    
    return {
        "authenticated": True,
        "api_key_hash": hash_api_key(credentials.credentials)[:8]  # First 8 chars for logging
    }