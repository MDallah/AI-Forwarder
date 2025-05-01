# app/core/config.py
import os
import logging
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
from dotenv import load_dotenv
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(' ' * 5 + os.path.basename(__file__) )

# Load .env file into environment variables BEFORE loading settings
# Useful especially when running scripts directly
load_dotenv(override=True)

class Settings(BaseSettings):
    # Backend API Keys
    openai_api_key: Optional[str] = Field(None, alias='OPENAI_API_KEY')
    anthropic_api_key: Optional[str] = Field(None, alias='ANTHROPIC_API_KEY')
    google_api_key: Optional[str] = Field(None, alias='GOOGLE_API_KEY')
    groq_api_key: Optional[str] = Field(None, alias='GROQ_API_KEY')  # Add Groq API key
    openrouter_api_key: Optional[str] = Field(None, alias='OPENROUTER_API_KEY')  # Add OpenRouter API key
    ollama_api_key: Optional[str] = Field(None, alias='OLLAMA_API_KEY')  # Add Ollama API key (optional for Ollama)

    # Forwarder keys (read only from file now)
    allowed_forwarder_keys: List[str] = []

    # Optional Base URLs
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    google_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    groq_base_url: str = "https://api.groq.com/openai/v1"  # Add Groq base URL
    openrouter_base_url: str = "https://openrouter.ai/api/v1"  # Add OpenRouter base URL
    ollama_base_url: str = "http://192.168.178.108:11434"  # Add Ollama base URL (default local)

    # Internal caches
    backend_api_keys: Dict[str, Optional[str]] = {}
    backend_base_urls: Dict[str, str] = {}

    @model_validator(mode='after')
    def process_settings(self) -> 'Settings':
        # Load forwarder keys from /keys/forwarder_api_keys.txt
        keys_file = Path(__file__).parent.parent.parent / "keys" / "forwarder_api_keys"
        if keys_file.exists():
            try:
                with keys_file.open("r") as f:
                    self.allowed_forwarder_keys = [line.strip() for line in f if line.strip()]
                    logger.info(f"Loaded {len(self.allowed_forwarder_keys)} forwarder keys from file.")
            except Exception as e:
                print(f"Error reading forwarder keys file: {e}")
                self.allowed_forwarder_keys = []
        else:
            print("WARNING: forwarder_api_keys.txt file not found. No keys loaded.")
            self.allowed_forwarder_keys = []

        # Backend keys and base URLs
        self.backend_api_keys = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google": self.google_api_key,
            "groq": self.groq_api_key,
            "openrouter": self.openrouter_api_key,
            "ollama": self.ollama_api_key,  # Add Ollama to backend API keys (may be None)
        }
        self.backend_base_urls = {
            "openai": self.openai_base_url,
            "anthropic": self.anthropic_base_url,
            "google": self.google_base_url,
            "groq": self.groq_base_url,
            "openrouter": self.openrouter_base_url,
            "ollama": self.ollama_base_url,  # Add Ollama to backend base URLs
        }

        if not any(self.backend_api_keys.values()):
            print("WARNING: No backend API keys set. Forwarding will fail.")

        return self

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore'

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
    # Add Groq provider
    # Groq's API structure groq:{model}
    "groq:": {
        "provider": "groq",
        "endpoint": "/chat/completions",
        "method": "POST",
        "auth_header": "Authorization",
        "auth_scheme": "Bearer",
    },
    # Add OpenRouter provider
    "openrouter:": {
        "provider": "openrouter",
        "endpoint": "/chat/completions",
        "method": "POST",
        "auth_header": "Authorization",
        "auth_scheme": "Bearer",
        "required_headers": {
            "HTTP-Referer": "https://api-forwarder.example.com",  # Should be customized
            "X-Title": "API Forwarder"  # Should be customized
        }
    },
    # Add Ollama provider
    "ollama:": {
        "provider": "ollama",
        "endpoint": "/api/chat",
        "method": "POST",
        "auth_header": "Authorization",  # Only used if API key is set
        "auth_scheme": "Bearer",
        "auth_optional": True,  # Indicates auth is optional for this provider
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
