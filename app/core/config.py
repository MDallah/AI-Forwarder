# app/core/config.py
import os
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
from dotenv import load_dotenv

# Load .env file into environment variables BEFORE loading settings
# Useful especially when running scripts directly
load_dotenv(override=True)

class Settings(BaseSettings):
    # Backend API Keys
    openai_api_key: Optional[str] = Field(None, alias='OPENAI_API_KEY')
    anthropic_api_key: Optional[str] = Field(None, alias='ANTHROPIC_API_KEY')
    google_api_key: Optional[str] = Field(None, alias='GOOGLE_API_KEY')
    # Add other keys here following the pattern: provider_api_key: Optional[str] = Field(None, alias='PROVIDER_API_KEY')

    # Forwarder Keys (keys allowed to access *this* service)
    # Loaded from FORWARDER_API_KEYS="key1,key2"
    forwarder_api_keys_str: str = Field("fwd_key_replace_me", alias='FORWARDER_API_KEYS') # Default added to avoid error if not set at all
    allowed_forwarder_keys: List[str] = []

    # Optional Base URLs
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    google_base_url: str = "https://generativelanguage.googleapis.com/v1beta" # Check latest for Gemini
    # Add other base URLs here

    # Store backend keys and base URLs in dictionaries for easier access
    backend_api_keys: Dict[str, Optional[str]] = {}
    backend_base_urls: Dict[str, str] = {}

    @model_validator(mode='after')
    def process_settings(self) -> 'Settings':
        # Process comma-separated forwarder keys
        if self.forwarder_api_keys_str:
            self.allowed_forwarder_keys = [key.strip() for key in self.forwarder_api_keys_str.split(',') if key.strip()]
        else:
            print("WARNING: No FORWARDER_API_KEYS defined in .env. The forwarder will not be accessible.")
            self.allowed_forwarder_keys = []

        # Populate backend key dictionary
        self.backend_api_keys = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google": self.google_api_key,
            # Add other providers here
        }
        # Populate base URL dictionary
        self.backend_base_urls = {
            "openai": self.openai_base_url,
            "anthropic": self.anthropic_base_url,
            "google": self.google_base_url,
            # Add other providers here
        }

        # Basic validation: Ensure at least one backend key is set
        if not any(self.backend_api_keys.values()):
            print("WARNING: No backend AI provider API keys (e.g., OPENAI_API_KEY) found in .env. Forwarding will fail.")
        
        # Remove the default key if real keys were loaded
        if "fwd_key_replace_me" in self.allowed_forwarder_keys and len(self.allowed_forwarder_keys) > 1:
             self.allowed_forwarder_keys.remove("fwd_key_replace_me")
        elif self.forwarder_api_keys_str == "fwd_key_replace_me":
            print("WARNING: Using default placeholder 'fwd_key_replace_me'. Please generate and add real keys to FORWARDER_API_KEYS in .env")


        return self

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore' # Ignore extra fields from .env

settings = Settings()

# Example provider mapping (can be expanded)
# Maps model prefixes to provider details
PROVIDER_MAPPING: Dict[str, Dict[str, Any]] = {
    "gpt-": {
        "provider": "openai",
        "endpoint": "/chat/completions",
        "method": "POST",
        "auth_header": "Authorization",
        "auth_scheme": "Bearer",
    },
    "claude-": {
        "provider": "anthropic",
        "endpoint": "/messages",
        "method": "POST",
        "auth_header": "x-api-key",
        "auth_scheme": None, # Key goes directly in header value
        "required_headers": { # Anthropic specific headers
            "anthropic-version": "2023-06-01"
            }
    },
    "gemini-": {
        "provider": "google",
        # Google's API structure is different (models/{model}:generateContent)
        "endpoint": "/models/{model}:generateContent", # Placeholder, needs model inserted
        "method": "POST",
        "auth_header": "x-goog-api-key",
        "auth_scheme": None, # Key goes directly in header value
    },
    # Add mappings for other providers (Cohere, Mistral, etc.)
}

def get_provider_info(model_name: str) -> Optional[Dict[str, Any]]:
    """Gets provider details based on the model name prefix."""
    for prefix, info in PROVIDER_MAPPING.items():
        if model_name.startswith(prefix):
            # Return a copy to avoid modifying the original mapping
            provider_info = info.copy()
            # Special handling for Google's endpoint format
            if provider_info["provider"] == "google":
                # Ensure the model name doesn't include "models/" prefix already
                clean_model_name = model_name.split('/')[-1]
                provider_info["endpoint"] = provider_info["endpoint"].format(model=clean_model_name)
            return provider_info
    return None
