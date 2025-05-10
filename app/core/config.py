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
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)
logger.info(f"Attempted to load environment variables from: {env_path}")


class Settings(BaseSettings):
    # Backend API Keys
    openai_api_key: Optional[str] = Field(None, alias='OPENAI_API_KEY')
    anthropic_api_key: Optional[str] = Field(None, alias='ANTHROPIC_API_KEY')
    google_api_key: Optional[str] = Field(None, alias='GOOGLE_API_KEY')
    groq_api_key: Optional[str] = Field(None, alias='GROQ_API_KEY')
    openrouter_api_key: Optional[str] = Field(None, alias='OPENROUTER_API_KEY')
    ollama_api_key: Optional[str] = Field(None, alias='OLLAMA_API_KEY')

    # Forwarder keys (read only from file now)
    allowed_forwarder_keys: List[str] = []

    # Optional Base URLs
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    google_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    ollama_base_url: str = "http://localhost:11434"

    # --- Provider Fallback Configuration ---
    providers_priority_str: Optional[str] = Field(None, alias='PROVIDERS_PRIORITY')
    openai_model: Optional[str] = Field(None, alias='OPENAI_MODEL')
    anthropic_model: Optional[str] = Field(None, alias='ANTHROPIC_MODEL')
    google_model: Optional[str] = Field(None, alias='GOOGLE_MODEL')
    groq_model: Optional[str] = Field(None, alias='GROQ_MODEL')
    openrouter_model: Optional[str] = Field(None, alias='OPENROUTER_MODEL')
    ollama_model: Optional[str] = Field(None, alias='OLLAMA_MODEL')

    # Internal caches
    backend_api_keys: Dict[str, Optional[str]] = {}
    backend_base_urls: Dict[str, str] = {}
    provider_priority_list: List[str] = []
    default_models: Dict[str, Optional[str]] = {}


    @model_validator(mode='after')
    def process_settings(self) -> 'Settings':
        # Load forwarder keys from /keys/forwarder_api_keys.txt
        # Use absolute path based on this file's location
        keys_file = Path(__file__).parent.parent.parent / "keys" / "forwarder_api_keys"
        if keys_file.exists():
            try:
                with keys_file.open("r") as f:
                    # Filter empty lines and potential comments starting with #
                    self.allowed_forwarder_keys = [
                        line.strip() for line in f if line.strip() and not line.strip().startswith('#')
                    ]
                logger.info(f"Loaded {len(self.allowed_forwarder_keys)} forwarder key(s) from: {keys_file}")
            except Exception as e:
                logger.error(f"Error reading forwarder keys file '{keys_file}': {e}", exc_info=True)
                self.allowed_forwarder_keys = []
        else:
            logger.warning(f"Forwarder keys file not found at: {keys_file}. No forwarder keys loaded.")
            self.allowed_forwarder_keys = []

        # Populate backend keys and base URLs dictionaries
        self.backend_api_keys = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google": self.google_api_key,
            "groq": self.groq_api_key,
            "openrouter": self.openrouter_api_key,
            "ollama": self.ollama_api_key,
        }
        self.backend_base_urls = {
            "openai": self.openai_base_url,
            "anthropic": self.anthropic_base_url,
            "google": self.google_base_url,
            "groq": self.groq_base_url,
            "openrouter": self.openrouter_base_url,
            "ollama": self.ollama_base_url,
        }

        # --- Process Provider Priority and Default Models ---
        if self.providers_priority_str:
            self.provider_priority_list = [p.strip().lower() for p in self.providers_priority_str.split(',') if p.strip()]
            logger.info(f"Provider priority list loaded: {self.provider_priority_list}")
        else:
            logger.info("PROVIDERS_PRIORITY not set. Fallback disabled.")
            self.provider_priority_list = []

        self.default_models = {
            "openai": self.openai_model,
            "anthropic": self.anthropic_model,
            "google": self.google_model,
            "groq": self.groq_model,
            "openrouter": self.openrouter_model,
            "ollama": self.ollama_model,
        }
        # Log which default models are set
        set_default_models = {k: v for k, v in self.default_models.items() if v}
        if set_default_models:
            logger.info(f"Default models configured: {set_default_models}")
        else:
            logger.info("No default provider models configured in .env (e.g., OPENAI_MODEL).")


        # --- Warnings ---
        if not any(self.backend_api_keys.values()):
             logger.warning("No backend provider API keys (e.g., OPENAI_API_KEY) seem to be configured in .env or environment.")
        if not self.allowed_forwarder_keys:
             logger.warning("No forwarder API keys loaded. Ensure 'keys/forwarder_api_keys' exists and contains keys, or run 'generate-key'.")
        elif self.allowed_forwarder_keys == ["fwd_key_replace_me"]: # Check for default placeholder
             logger.warning("Default placeholder 'fwd_key_replace_me' found in forwarder keys. Please generate and use real keys.")

        # Validate provider priority list against known providers
        valid_providers = set(PROVIDER_MAPPING.keys())
        invalid_priority = [p for p in self.provider_priority_list if f"{p}:" not in valid_providers and f"{p}-" not in valid_providers and p not in valid_providers]
        # Adjust check for prefix-based keys
        valid_priority_providers = set()
        for p in self.provider_priority_list:
             found = False
             for prefix in PROVIDER_MAPPING.keys():
                 # Check direct match or if provider name starts with prefix (e.g. 'openai' matches 'gpt-') - less precise but covers base case
                 provider_name_in_map = PROVIDER_MAPPING[prefix]['provider']
                 if p == provider_name_in_map:
                     valid_priority_providers.add(p)
                     found = True
                     break
             if not found:
                 logger.warning(f"Provider '{p}' in PROVIDERS_PRIORITY is not a known provider key in PROVIDER_MAPPING ({list(PROVIDER_MAPPING.keys())}). It will be ignored.")
        # Update the list to only contain valid ones recognised by PROVIDER_MAPPING logic
        self.provider_priority_list = [p for p in self.provider_priority_list if p in valid_priority_providers]
        if self.provider_priority_list:
             logger.info(f"Validated provider priority list: {self.provider_priority_list}")


        return self

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore'
        # Allow reading from environment variables as highest priority
        # Already default behavior in Pydantic v2

# --- PROVIDER_MAPPING (Ensure keys match provider names used elsewhere, e.g., 'openai', 'anthropic') ---
# Maps unique prefixes OR provider names to provider details
# IMPORTANT: The keys here ('openai', 'anthropic', etc.) must match the names
#            used in PROVIDERS_PRIORITY and the default model env vars (e.g., OPENAI_MODEL)
PROVIDER_MAPPING: Dict[str, Dict[str, Any]] = {
    # Unique prefix based matching (preferred for clarity)
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
        "endpoint": "/models/{model}:generateContent", # Placeholder, needs model inserted
        "method": "POST",
        "auth_header": "x-goog-api-key",
        "auth_scheme": None, # Key goes directly in header value
    },
    "groq:": { # Using ':' suffix for clarity
        "provider": "groq",
        "endpoint": "/chat/completions",
        "method": "POST",
        "auth_header": "Authorization",
        "auth_scheme": "Bearer",
    },
    "openrouter:": { # Using ':' suffix
        "provider": "openrouter",
        "endpoint": "/chat/completions",
        "method": "POST",
        "auth_header": "Authorization",
        "auth_scheme": "Bearer",
        "required_headers": {
             # Customize these as needed for OpenRouter identification
            # "HTTP-Referer": "YOUR_SITE_URL", # Optional: Replace with your site URL
            # "X-Title": "YOUR_APP_NAME" # Optional: Replace with your app name
        }
    },
     "ollama:": { # Using ':' suffix
        "provider": "ollama",
        "endpoint": "/api/chat", # OpenAI compatible endpoint is /v1/chat/completions if using Ollama's OpenAI compatibility layer
        "method": "POST",
        "auth_header": "Authorization", # Only used if OLLAMA_API_KEY is set
        "auth_scheme": "Bearer",
        "auth_optional": True, # Auth is typically optional for local Ollama
    },
     # --- Fallback mapping by *provider name* if no prefix matches ---
     # These allow matching providers listed in PROVIDERS_PRIORITY even if their
     # default model doesn't have a unique prefix defined above.
     "openai": {
         "provider": "openai",
         "endpoint": "/chat/completions",
         "method": "POST",
         "auth_header": "Authorization",
         "auth_scheme": "Bearer",
     },
     "anthropic": {
         "provider": "anthropic",
         "endpoint": "/messages",
         "method": "POST",
         "auth_header": "x-api-key",
         "auth_scheme": None,
         "required_headers": {"anthropic-version": "2023-06-01"}
     },
     "google": {
         "provider": "google",
         "endpoint": "/models/{model}:generateContent", # Still needs model formatting
         "method": "POST",
         "auth_header": "x-goog-api-key",
         "auth_scheme": None,
     },
     "groq": {
         "provider": "groq",
         "endpoint": "/chat/completions",
         "method": "POST",
         "auth_header": "Authorization",
         "auth_scheme": "Bearer",
     },
     "openrouter": {
         "provider": "openrouter",
         "endpoint": "/chat/completions",
         "method": "POST",
         "auth_header": "Authorization",
         "auth_scheme": "Bearer",
         "required_headers": {
             # Customize as needed
             # "HTTP-Referer": "YOUR_SITE_URL",
             # "X-Title": "YOUR_APP_NAME"
         }
     },
     "ollama": {
        "provider": "ollama",
        "endpoint": "/api/chat",
        "method": "POST",
        "auth_header": "Authorization",
        "auth_scheme": "Bearer",
        "auth_optional": True,
     }
}

# Initialize settings once
settings = Settings()
logger.info("Settings loaded successfully.") # Log successful loading


def get_provider_info(model_name: Optional[str] = None, provider_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Gets provider details based on the model name prefix OR the provider name.
    Prefers model name matching if provided.
    """
    provider_info = None
    used_key = None # Track which key was used for matching

    # 1. Try matching by model name prefix (if model_name is provided)
    if model_name:
        for prefix, info in PROVIDER_MAPPING.items():
            # Ensure prefix has content and check startswith
            if prefix and model_name.startswith(prefix):
                provider_info = info.copy() # Return a copy
                used_key = prefix
                # logger.debug(f"Found provider info for model '{model_name}' using prefix '{prefix}'")
                break # Found best match

    # 2. If no prefix match OR model_name wasn't provided, try matching by provider name (if provider_name is provided)
    if not provider_info and provider_name:
        provider_name_lower = provider_name.lower()
        # Check if the provider name itself is a key in the mapping
        if provider_name_lower in PROVIDER_MAPPING:
             # Check if the mapping key directly matches the provider name
             if PROVIDER_MAPPING[provider_name_lower].get("provider") == provider_name_lower:
                provider_info = PROVIDER_MAPPING[provider_name_lower].copy()
                used_key = provider_name_lower
                # logger.debug(f"Found provider info using provider name '{provider_name_lower}'")


    # Special handling (needs to happen *after* finding the base info)
    if provider_info:
        actual_provider = provider_info.get("provider")
        # Handle Google's dynamic endpoint format
        if actual_provider == "google" and model_name:
            # Ensure the model name doesn't include "models/" prefix already
            clean_model_name = model_name.split('/')[-1]
            if '{model}' in provider_info["endpoint"]: # Check if formatting is needed
                 provider_info["endpoint"] = provider_info["endpoint"].format(model=clean_model_name)
            else: # If endpoint was already formatted somehow, log a warning
                 logger.warning(f"Google endpoint for key '{used_key}' seems pre-formatted: {provider_info['endpoint']}. Expected '{{model}}' placeholder.")
        elif not model_name and provider_info["provider"] == "google":
             logger.error("Cannot format Google endpoint without a specific model name.")
             return None # Cannot proceed without a model for Google

    if not provider_info:
         logger.warning(f"Could not determine provider info for model='{model_name}', provider='{provider_name}'")

    return provider_info