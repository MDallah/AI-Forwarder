# app/core/forwarder.py
import os
import httpx
import json
import logging
from typing import Dict, Any, Tuple, Optional, List
from .config import settings, get_provider_info
from app.models.api import ChatCompletionRequest, ForwarderResponse, ChatMessage

logging.basicConfig(level=logging.INFO)
# Use specific logger name
logger = logging.getLogger(' ' * 5 + os.path.basename(__file__))


async def forward_request_to_provider(
    original_request: ChatCompletionRequest,
    client: httpx.AsyncClient
) -> ForwarderResponse:
    """
    Forwards the request to the appropriate AI provider based on the model name.
    """
    model_name = original_request.model
    if not model_name:
         # This should ideally be caught before calling this function in fallback mode,
         # but handle defensively.
         logger.error("forward_request_to_provider called without a model name.")
         return ForwarderResponse(success=False, error="Internal error: No model name provided for forwarding.")

    # Use the enhanced get_provider_info which can also use provider name if needed
    provider_info = get_provider_info(model_name=model_name)

    if not provider_info:
        # If no info found based on model prefix, maybe it's just a provider name? Unlikely here.
        logger.warning(f"No provider mapping found for model prefix/name: {model_name}")
        return ForwarderResponse(success=False, error=f"Unsupported model or provider configuration for: {model_name}", model_used=model_name)

    provider = provider_info["provider"]
    backend_api_key = settings.backend_api_keys.get(provider)
    base_url = settings.backend_base_urls.get(provider)
    endpoint = provider_info["endpoint"] # This might already be formatted for Google by get_provider_info
    method = provider_info["method"]
    auth_header_name = provider_info["auth_header"]
    auth_scheme = provider_info.get("auth_scheme") # Optional (like Bearer)
    required_headers = provider_info.get("required_headers", {})
    auth_optional = provider_info.get("auth_optional", False)  # Check if auth is optional

    # Check for API Key *unless* auth is optional for this provider
    if not backend_api_key and not auth_optional:
        logger.error(f"API key for provider '{provider}' is required but not configured.")
        return ForwarderResponse(success=False, provider=provider, model_used=model_name, error=f"API key for provider '{provider}' is not configured.")
    if not base_url:
        logger.error(f"Base URL for provider '{provider}' is not configured.")
        return ForwarderResponse(success=False, provider=provider, model_used=model_name, error=f"Base URL for provider '{provider}' is not configured.")

    # --- Prepare Headers ---
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    headers.update(required_headers) # Add provider-specific required headers

    # Set Authorization header if API key is available OR if auth is not optional (even if key is None, might be intended)
    # Only skip auth header entirely if auth is optional AND no key is provided.
    if backend_api_key or not auth_optional:
        if backend_api_key: # Only include key if it exists
            auth_value = f"{auth_scheme} {backend_api_key}" if auth_scheme else backend_api_key
            headers[auth_header_name] = auth_value
        elif not auth_optional:
             # This case should have been caught above, but log defensively.
             logger.error(f"Auth required for {provider} but key is missing (Header '{auth_header_name}' will be missing).")
             # Return error here as request will fail anyway
             return ForwarderResponse(success=False, provider=provider, model_used=model_name, error=f"API key for provider '{provider}' is not configured.")


    # --- Prepare Request Body ---
    try:
        payload = adapt_request_payload(provider, original_request)
    except ValueError as e:
            logger.error(f"Failed to adapt request payload for {provider}: {e}", exc_info=True)
            return ForwarderResponse(success=False, provider=provider, model_used=model_name, error=f"Failed to adapt request payload: {e}")
    except Exception as e: # Catch broader errors during adaptation
            logger.error(f"Unexpected error adapting request payload for {provider}: {e}", exc_info=True)
            return ForwarderResponse(success=False, provider=provider, model_used=model_name, error=f"Unexpected error adapting payload: {e}")


    # --- Make the Request ---
    target_url = f"{base_url.rstrip('/')}{endpoint}"
    # Sanity check URL
    if not target_url.startswith(("http://", "https://")):
         logger.error(f"Invalid target URL constructed for {provider}: {target_url}")
         return ForwarderResponse(success=False, provider=provider, model_used=model_name, error=f"Internal configuration error: Invalid target URL '{target_url}'.")

    try:
        # Use the provider's actual model name in logs if different (e.g. stripped prefix)
        actual_model_in_payload = payload.get("model", model_name) # Get model from payload if exists
        logger.info(f"Forwarding to {provider}: URL={target_url}, Request_Model={model_name}, Payload_Model={actual_model_in_payload}")

        # DEBUG: Log payload and headers carefully if needed
        # try:
        #     payload_str = json.dumps(payload)
        #     if len(payload_str) < 1000: # Avoid logging huge payloads
        #         logger.debug(f"Payload for {provider}: {payload_str}")
        #     else:
        #         logger.debug(f"Payload for {provider} (truncated): {payload_str[:500]}...")
        # except TypeError:
        #     logger.debug(f"Payload for {provider} (not json serializable): {payload}")
        # Sensitive header logging - mask auth keys
        # masked_headers = {k: (v[:10] + '...' if k.lower() in ['authorization', 'x-api-key', 'x-goog-api-key'] and isinstance(v, str) else v) for k, v in headers.items()}
        # logger.debug(f"Headers for {provider}: {masked_headers}")

        response = await client.request(
            method=method,
            url=target_url,
            headers=headers,
            json=payload,
            # Timeout is now set on the client instance in main.py lifespan
        )

        # Log response status before potential raise_for_status
        logger.info(f"Received response from {provider}: Status Code={response.status_code}")
        # DEBUG: Log raw response if needed (careful with large responses)
        # if response.status_code >= 400 or logger.isEnabledFor(logging.DEBUG):
        #      logger.debug(f"Raw Response Text (first 500 chars): {response.text[:500]}")

        response.raise_for_status() # Raise exception for 4xx/5xx errors

        # Attempt to parse JSON before declaring success
        try:
            response_data = response.json()
            return ForwarderResponse(
                success=True,
                data=response_data,
                provider=provider,
                model_used=model_name # Report the model requested by the user/fallback logic
            )
        except json.JSONDecodeError as e:
            error_msg = f"Forwarding to {provider} succeeded ({response.status_code}) but failed to decode JSON response: {e}. Response text: {response.text[:500]}..."
            logger.error(error_msg)
            # Return success=False as we can't return valid data
            return ForwarderResponse(
                success=False,
                provider=provider,
                model_used=model_name,
                error=error_msg,
                status_code=response.status_code, # Keep original status code
                error_details=response.text # Store raw text as details
                )

    except httpx.TimeoutException as e:
        error_msg = f"Request timed out requesting {provider}: {type(e).__name__} - {e}"
        logger.error(error_msg)
        return ForwarderResponse(success=False, provider=provider, model_used=model_name, error=error_msg, status_code=status.HTTP_504_GATEWAY_TIMEOUT)
    except httpx.RequestError as e:
        # Covers connection errors, DNS errors etc.
        error_msg = f"Network error requesting {provider}: {type(e).__name__} - {e}"
        logger.error(error_msg)
        # Use 503 Service Unavailable for network issues connecting to backend
        return ForwarderResponse(success=False, provider=provider, model_used=model_name, error=error_msg, status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
    except httpx.HTTPStatusError as e:
        error_short_msg = f"Error from {provider} ({e.response.status_code})"
        logger.warning(f"{error_short_msg}. Response body: {e.response.text[:500]}...") # Log start of error body
        parsed_error_details = None
        full_error_msg = f"{error_short_msg}"
        try:
            # Try parsing the error response from the provider
            parsed_error_details = e.response.json()
            # Append more detail if parsing is successful
            # Example: extract a message field if common across providers
            provider_message = parsed_error_details.get("error", {}).get("message") or parsed_error_details.get("detail")
            if provider_message and isinstance(provider_message, str):
                 full_error_msg += f": {provider_message}"
            # Alternatively, just append the whole parsed structure (can be verbose)
            # full_error_msg += f": {parsed_error_details}"
        except json.JSONDecodeError:
            # If JSON parsing fails, use raw text
            parsed_error_details = e.response.text
            full_error_msg += f": {e.response.text[:200]}" # Append truncated raw text

        return ForwarderResponse(
            success=False,
            provider=provider,
            model_used=model_name,
            error=full_error_msg, # More descriptive error
            status_code=e.response.status_code,
            error_details=parsed_error_details # Store details (parsed JSON or raw text)
            )
    # Removed JSONDecodeError here as it's handled after successful status code check
    except Exception as e:
        error_msg = f"An unexpected error occurred during forwarding to {provider}: {type(e).__name__} - {e}"
        logger.exception(error_msg) # Log full traceback for unexpected errors
        return ForwarderResponse(success=False, provider=provider, model_used=model_name, error=error_msg, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


def adapt_request_payload(provider: str, request: ChatCompletionRequest) -> Dict[str, Any]:
    """
    Adapts the incoming ChatCompletionRequest format to the target provider's expected format.
    """
    payload: Dict[str, Any] = {}
    model_name = request.model # Get the model name passed for this specific attempt

    # --- Provider-Specific Model Name Handling ---
    # Strip prefixes used for routing if the provider API expects the base name
    provider_model_name = model_name
    if provider == "groq" and model_name.startswith("groq:"):
        provider_model_name = model_name.split(":", 1)[1]
    elif provider == "openrouter" and model_name.startswith("openrouter:"):
        provider_model_name = model_name.split(":", 1)[1]
    elif provider == "ollama" and model_name.startswith("ollama:"):
        provider_model_name = model_name.split(":", 1)[1]
    # Google model name is handled by endpoint formatting in get_provider_info
    elif provider == "google":
        pass # Model name not needed in Google payload
    else:
        # For OpenAI, Anthropic, etc., use the model name as is (assuming it doesn't have a prefix)
        pass

    # Set model in payload ONLY if the provider requires it (most do, except Google)
    if provider not in ["google"]:
        payload["model"] = provider_model_name

    # --- Common Parameters ---
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        # Check provider compatibility (e.g., Anthropic doesn't support top_p in messages API)
        if provider != "anthropic":
            payload["top_p"] = request.top_p

    # Handle max_tokens
    if request.max_tokens is not None:
        if provider == "anthropic":
            payload["max_tokens"] = request.max_tokens # v3 API uses max_tokens
        elif provider == "google":
            pass # Handled in generationConfig later
        else: # Assume OpenAI/Groq/OpenRouter/Ollama -like
            payload["max_tokens"] = request.max_tokens

    # --- Messages/Prompts Adaptation ---
    if provider == "openai" or provider == "groq" or provider == "openrouter":
        # OpenAI, Groq, OpenRouter use the standard messages format
        payload["messages"] = [msg.model_dump(exclude_none=True) for msg in request.messages]

    elif provider == "anthropic":
        # Anthropic v3 API structure
        system_prompt: Optional[str] = None
        anthropic_messages: List[Dict[str, Any]] = []
        for msg in request.messages:
            if msg.role == "system" and msg.content: # Ensure content exists for system prompt
                # Anthropic expects only one system message
                if system_prompt is None:
                    system_prompt = msg.content
                else:
                    logger.warning("Multiple system messages found for Anthropic request, using the first one.")
            elif msg.role in ["user", "assistant"] and msg.content is not None: # Must have content
                anthropic_messages.append({"role": msg.role, "content": msg.content})
            elif msg.content is None and msg.role != "system":
                 logger.warning(f"Skipping message with null content for role '{msg.role}' in Anthropic request.")


        # Validate message alternation if possible (or let Anthropic API handle it)
        # TODO: Add validation for strictly alternating user/assistant roles if needed

        payload["messages"] = anthropic_messages
        if system_prompt:
            payload["system"] = system_prompt
        # Remove params maybe not supported by Anthropic base
        # payload.pop("top_p", None) # Already handled above

    elif provider == "google":
        # Google Gemini API structure ('contents')
        role_map = {"user": "user", "assistant": "model"} # System handled separately

        contents = []
        system_instruction: Optional[Dict[str, Any]] = None

        # Google recommends putting system instructions first if possible
        system_messages = [msg for msg in request.messages if msg.role == "system" and msg.content]
        other_messages = [msg for msg in request.messages if msg.role != "system"]

        if system_messages:
            # Combine multiple system messages if necessary, or use the first one
            system_content = "\n".join([msg.content for msg in system_messages])
            # Simple approach: Add as a 'user' role part at the beginning of 'contents'
            # More complex: Use 'system_instruction' field if supported by the specific model/API version
            # For now, prepend as user content
            # contents.append({"role": "user", "parts": [{"text": system_content}]})
            # Let's try the dedicated field (check Gemini API docs for model support)
            system_instruction = {"role": "system", "parts": [{"text": system_content}]}
            # Note: This might need to be {"parts": [...]} directly under a top-level "systemInstruction" key
            # Checking API reference: It's a top-level `system_instruction` object containing `parts`.
            payload["system_instruction"] = {"parts": [{"text": system_content}]} # Correct structure

        last_role = None
        current_parts = []
        for msg in other_messages:
             if msg.content: # Ensure content exists
                mapped_role = role_map.get(msg.role)
                if not mapped_role:
                     logger.warning(f"Skipping message with unsupported role '{msg.role}' for Google provider.")
                     continue

                # Google requires alternating roles (user/model). If consecutive messages
                # have the same mapped role, combine their content into 'parts' of a single content object.
                if mapped_role == last_role:
                     current_parts.append({"text": msg.content}) # Append to existing parts
                else:
                     # If role changed, finalize the previous content object (if any)
                     if current_parts:
                         contents.append({"role": last_role, "parts": current_parts})
                     # Start a new content object
                     current_parts = [{"text": msg.content}]
                     last_role = mapped_role

        # Append the last collected parts
        if current_parts:
            contents.append({"role": last_role, "parts": current_parts})


        payload["contents"] = contents

        # Adapt other parameters into generationConfig
        generation_config = {}
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.max_tokens is not None:
            generation_config["maxOutputTokens"] = request.max_tokens
        if request.top_p is not None:
            generation_config["topP"] = request.top_p
        if request.stop:
             # Google expects a list of strings
             stop_sequences = request.stop if isinstance(request.stop, list) else [request.stop]
             generation_config["stopSequences"] = stop_sequences

        if generation_config:
            payload["generationConfig"] = generation_config

        # Remove core parameters that are now in generationConfig or handled via URL
        payload.pop("temperature", None)
        payload.pop("max_tokens", None)
        payload.pop("top_p", None)
        payload.pop("stop", None)


    elif provider == "ollama":
        # Ollama uses OpenAI-like '/api/chat' structure (or '/v1/chat/completions' if using compat endpoint)
        payload["messages"] = [msg.model_dump(exclude_none=True) for msg in request.messages]
        # Keep parameters like temperature, max_tokens if they exist in payload
        # Ollama specific: 'options' dictionary for parameters like temperature, top_p, num_predict (max_tokens)
        options = {}
        if "temperature" in payload:
            options["temperature"] = payload.pop("temperature")
        if "top_p" in payload:
             options["top_p"] = payload.pop("top_p")
        if "max_tokens" in payload:
             # Ollama uses 'num_predict' for max tokens
             options["num_predict"] = payload.pop("max_tokens")
        if request.stop:
             options["stop"] = request.stop if isinstance(request.stop, list) else [request.stop]
             payload.pop("stop", None) # Remove from top level if present

        if options:
             payload["options"] = options

        # Ollama requires stream: false for non-streaming requests via /api/chat
        payload["stream"] = False

    else:
        # Should not be reached if provider mapping is correct
        logger.error(f"Payload adaptation not implemented for unknown provider: {provider}")
        raise ValueError(f"Payload adaptation logic missing for provider '{provider}'.")

    # Add any provider-specific parameters from extra_body
    if request.extra_body:
        logger.info(f"Applying extra_body parameters for {provider}: {list(request.extra_body.keys())}")
        # Merge carefully, potentially overwriting adapted params if specified in extra_body
        payload.update(request.extra_body)

    # Remove stream parameter unless provider specifically handles it (like Ollama)
    if provider != "ollama" and "stream" in payload:
        # Only remove if it wasn't part of extra_body? For now, always remove if not Ollama.
        payload.pop("stream", None)


    return payload