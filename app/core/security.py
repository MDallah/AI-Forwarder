# app/core/security.py
import secrets
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from .config import settings

API_KEY_HEADER = APIKeyHeader(name="Authorization", auto_error=False) # Set auto_error=False for custom handling
FORWARDER_KEY_PREFIX = "fwd_" # Recommended prefix for clarity

def generate_api_key(length: int = 32) -> str:
    """Generates a secure, random API key."""
    return FORWARDER_KEY_PREFIX + secrets.token_urlsafe(length)

async def validate_forwarder_key(api_key: str = Depends(API_KEY_HEADER)) -> str:
    """Dependency to validate the incoming forwarder API key."""
    if not api_key:
         raise HTTPException(
             status_code=status.HTTP_401_UNAUTHORIZED,
             detail="Authorization header is missing",
         )

    if not api_key.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected 'Bearer <key>'",
        )

    key = api_key.split(" ", 1)[1] # Split only once

    if key not in settings.allowed_forwarder_keys:
        print(f"Failed auth attempt with key: {key[:5]}...") # Log redacted key attempt
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or unauthorized API Key",
        )
    return key # Return the valid key if needed elsewhere
