# app/core/forwarder.py
import httpx
import json
import logging
from typing import Dict, Any, Tuple, Optional, List
from .config import settings, get_provider_info
from app.models.api import ChatCompletionRequest, ForwarderResponse, ChatMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def forward_request_to_provider(
    original_request: ChatCompletionRequest,
    client: httpx.AsyncClient
) -> ForwarderResponse:
    """
    Forwards the request to the appropriate AI provider based on the model name.
    """
    model_name = original_request.model
    provider_info = get_provider_info(model_name)

    if not provider_info:
        return ForwarderResponse(success=False, error=f"Unsupported model prefix or provider for: {model_name}", model_used=model_name)

    provider = provider_info["provider"]
    backend_api_key = settings.backend_api_keys.get(provider)
    base_url = settings.backend_base_urls.get(provider)
    endpoint = provider_info["endpoint"] # This might already be formatted for Google
    method = provider_info["method"]
    auth_header_name = provider_info["auth_header"]
    auth_scheme = provider_info.get("auth_scheme") # Optional (like Bearer)
    required_headers = provider_info.get("required_headers", {})

    if not backend_api_key:
        return ForwarderResponse(success=False, provider=provider, model_used=model_name, error=f"API key for provider '{provider}' is not configured.")
    if not base_url:
            return ForwarderResponse(success=False, provider=provider, model_used=model_name, error=f"Base URL for provider '{provider}' is not configured.")

    # --- Prepare Headers ---
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    headers.update(required_headers) # Add provider-specific required headers

    # Set Authorization header
    if auth_scheme:
        headers[auth_header_name] = f"{auth_scheme} {backend_api_key}"
    else:
        headers[auth_header_name] = backend_api_key # Key directly in value

    # --- Prepare Request Body ---
    try:
        payload = adapt_request_payload(provider, original_request)
    except ValueError as e:
            return ForwarderResponse(success=False, provider=provider, model_used=model_name, error=f"Failed to adapt request payload: {e}")


    # --- Make the Request ---
    target_url = f"{base_url.rstrip('/')}{endpoint}"
    try:
        logger.info(f"Forwarding to {provider}: URL={target_url}, Model={model_name}") # Logging
        # logger.debug(f"Payload: {json.dumps(payload, indent=2)}") # Careful logging payload
        # logger.debug(f"Headers: {headers}") # Careful logging headers

        response = await client.request(
            method=method,
            url=target_url,
            headers=headers,
            json=payload,
            timeout=120.0, # Increase timeout for potentially long generations
        )

        # Log response status before potential raise_for_status
        logger.info(f"Received response from {provider}: Status Code={response.status_code}")
        # logger.debug(f"Raw Response: {response.text}") # Log raw response if needed

        response.raise_for_status() # Raise exception for 4xx/5xx errors

        # Return the successful response data directly
        return ForwarderResponse(
            success=True,
            data=response.json(),
            provider=provider,
            model_used=model_name
        )

    except httpx.RequestError as e:
        error_msg = f"Network error requesting {provider}: {type(e).__name__} - {e}"
        logger.error(error_msg)
        return ForwarderResponse(success=False, provider=provider, model_used=model_name, error=error_msg)
    except httpx.HTTPStatusError as e:
        error_msg = f"Error from {provider} ({e.response.status_code})"
        logger.warning(f"{error_msg}. Response body: {e.response.text[:500]}...") # Log start of error body
        parsed_error = None
        try:
            parsed_error = e.response.json()
            error_msg += f": {parsed_error}" # Add parsed JSON detail if possible
        except json.JSONDecodeError:
            error_msg += f": {e.response.text}" # Add raw text if not JSON

        return ForwarderResponse(
            success=False,
            provider=provider,
            model_used=model_name,
            error=f"Error from {provider} ({e.response.status_code})",
            status_code=e.response.status_code,
            error_details=parsed_error or e.response.text # Store details
            )
    except json.JSONDecodeError as e:
        error_msg = f"Failed to decode JSON response from {provider}: {e}. Response text: {response.text[:500]}..."
        logger.error(error_msg)
        return ForwarderResponse(success=False, provider=provider, model_used=model_name, error=error_msg, data=response.text)
    except Exception as e:
        error_msg = f"An unexpected error occurred during forwarding: {type(e).__name__} - {e}"
        logger.exception(error_msg) # Log full traceback for unexpected errors
        return ForwarderResponse(success=False, provider=provider, model_used=model_name, error=error_msg)

def adapt_request_payload(provider: str, request: ChatCompletionRequest) -> Dict[str, Any]:
    """
    Adapts the incoming ChatCompletionRequest format to the target provider's expected format.
    This needs significant expansion based on provider API differences.
    """
    # Start with core fields that are often similar or need slight renaming
    payload: Dict[str, Any] = {}
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p # Check provider compatibility

    # Handle model field (sometimes in payload, sometimes not)
    # payload["model"] = request.model # Often needed, but not for Google

    # Handle max_tokens (common variations)
    if request.max_tokens is not None:
        if provider == "anthropic":
            payload["max_tokens"] = request.max_tokens # v3 API uses max_tokens
        elif provider == "google":
            # Handled in generationConfig later
            pass
        else: # Assume OpenAI-like
            payload["max_tokens"] = request.max_tokens

    # Handle messages/prompts (major variations)
    if provider == "openai":
        payload["model"] = request.model # OpenAI needs model in payload
        payload["messages"] = [msg.model_dump(exclude_none=True) for msg in request.messages]

    elif provider == "anthropic":
        # Anthropic v3 API structure
        payload["model"] = request.model # Anthropic needs model
        system_prompt: Optional[str] = None
        anthropic_messages: List[Dict[str, Any]] = []
        for msg in request.messages:
            if msg.role == "system":
                system_prompt = msg.content
            else: # user, assistant
                # Anthropic expects strictly alternating user/assistant messages
                # This basic adaptation assumes the input follows this or handles simple cases.
                # More complex validation/reordering might be needed for robust handling.
                if msg.content is not None: # Skip messages with null content potentially?
                    anthropic_messages.append({"role": msg.role, "content": msg.content})

        payload["messages"] = anthropic_messages
        if system_prompt:
            payload["system"] = system_prompt
        # Remove params maybe not supported by Anthropic base
        payload.pop("top_p", None)

    elif provider == "google":
        # Google Gemini API structure ('contents')
        # Maps OpenAI roles to Gemini roles
        role_map = {"user": "user", "assistant": "model", "system": "user"} # Often map system to first user turn

        contents = []
        system_instruction: Optional[Dict[str, Any]] = None

        for msg in request.messages:
            # Google GenAI standard: System prompt as separate object
            if msg.role == "system" and msg.content:
                 # Check if gemini supports system_instruction object
                 # For now, prepend as a user message part or handle via specific logic if needed.
                 # Simplified: Treat system as first user message part
                 mapped_role = "user"
                 if msg.content:
                     contents.append({"role": mapped_role, "parts": [{"text": msg.content}]})

            elif msg.content: # Ensure content exists
                mapped_role = role_map.get(msg.role, "user") # Default to user if unknown role
                # Check if last message was the same role, if so, append parts (Google requires alternating roles)
                # This simplification assumes input is already correctly alternated or model handles it.
                contents.append({"role": mapped_role, "parts": [{"text": msg.content}]})

        payload["contents"] = contents

        # Adapt other parameters into generationConfig
        generation_config = {}
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.max_tokens is not None:
            generation_config["maxOutputTokens"] = request.max_tokens
        if request.top_p is not None:
            generation_config["topP"] = request.top_p
        # Add stop sequences if needed: generation_config["stopSequences"] = request.stop if request.stop else []

        if generation_config:
            payload["generationConfig"] = generation_config

        # Google model name is part of the URL, not payload
        
    elif provider == "groq":
        # Groq uses OpenAI-compatible API
        payload["model"] = request.model.split(":")[1]
        payload["messages"] = [msg.model_dump(exclude_none=True) for msg in request.messages]
        
        # Add any Groq-specific parameters here if needed
        # For now, Groq is fully compatible with OpenAI's format

    else:
        # Default or other providers: Assume OpenAI-like for now
        payload["model"] = request.model
        payload["messages"] = [msg.model_dump(exclude_none=True) for msg in request.messages]


    # Add any provider-specific parameters from extra_body
    if request.extra_body:
        logger.info(f"Applying extra_body parameters for {provider}: {request.extra_body.keys()}")
        payload.update(request.extra_body)

    # Remove stream parameter for now (or handle streaming separately)
    payload.pop("stream", None) # Remove stream if present, as we don't support it yet

    return payload
