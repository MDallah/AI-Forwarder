# app/main.py
import os
import typer
import uvicorn
import httpx
import json
import logging
import copy # Needed for deep copying request object
from typing import Optional, List, Tuple # Added List, Tuple
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pathlib import Path # Use pathlib for safer path joining

from app.core import config, security
from app.core.forwarder import forward_request_to_provider
from app.models.api import ChatCompletionRequest, ForwarderResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(' ' * 5 + os.path.basename(__file__))

# --- Async Lifecycle for HTTPX client ---
# Create a global httpx client to reuse connections
lifespan_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global lifespan_client
    # Startup: Initialize the HTTPX client
    # Increase default timeouts for potentially long API calls
    timeout = httpx.Timeout(10.0, read=120.0) # 10s connect, 120s read
    lifespan_client = httpx.AsyncClient(timeout=timeout)
    logger.info(f"HTTPX Client started with timeout: {timeout}.")
    yield
    # Shutdown: Close the HTTPX client
    if lifespan_client:
        await lifespan_client.aclose()
        logger.info("HTTPX Client closed.")

# --- FastAPI App Setup ---
app = FastAPI(
    title="AI Forwarder",
    description="Forwards requests to various AI providers using stored API keys. Supports provider fallback.",
    version="0.3.0", # Updated version
    lifespan=lifespan # Use the lifespan context manager
)

# --- Mount Static Files & Templates ---
# Get the directory of the current file (main.py)
current_dir = Path(__file__).parent
base_dir = current_dir.parent # Go up one level to the project root (where 'static' and 'app' are)
static_dir = base_dir / "static"
templates_dir = current_dir / "templates"

templates = None
try:
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        logger.info(f"Mounted static files from: {static_dir}")
    else:
         logger.warning(f"Static directory not found at: {static_dir}. Frontend may not load correctly.")

    if templates_dir.is_dir():
        templates = Jinja2Templates(directory=templates_dir)
        logger.info(f"Loaded HTML templates from: {templates_dir}")
    else:
        logger.warning(f"Templates directory not found at: {templates_dir}. Root ('/') endpoint will fail.")

except Exception as e:
     logger.error(f"Error mounting static files or templates: {e}", exc_info=True)


# --- Helper Function ---
def _get_provider_status() -> dict:
    """Helper to get backend provider configuration status."""
    status_dict = {}
    for provider, key in config.settings.backend_api_keys.items():
        # Only show providers relevant to the mapping keys (like 'openai', 'anthropic', etc.)
        # Check if provider name exists in any of the mapping details
        is_relevant = any(info.get("provider") == provider for info in config.PROVIDER_MAPPING.values())
        if is_relevant:
            is_configured = bool(key) or config.PROVIDER_MAPPING.get(provider, {}).get("auth_optional", False)
            default_model = config.settings.default_models.get(provider)
            status_dict[provider] = {
                "configured": is_configured,
                "default_model": default_model if default_model else "Not Set"
            }
    return status_dict

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root(request: Request):
    if not templates:
        return HTMLResponse("<html><body><h1>AI Forwarder</h1><p>Error: HTML Templates directory not found or not loaded. Check server logs.</p></body></html>", status_code=500)
    # Pass info to the template
    provider_status = _get_provider_status()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "AI Forwarder Status",
        "forwarder_keys_count": len(config.settings.allowed_forwarder_keys),
        "provider_status": provider_status,
        "provider_priority": config.settings.provider_priority_list or ["Not Configured"],
        "api_docs_url": app.docs_url or "/docs", # Provide defaults if None
        "redoc_url": app.redoc_url or "/redoc"
    })

# Determine which status codes from backend should trigger fallback
RETRYABLE_STATUS_CODES = {
    status.HTTP_429_TOO_MANY_REQUESTS,
    status.HTTP_500_INTERNAL_SERVER_ERROR,
    status.HTTP_502_BAD_GATEWAY,
    status.HTTP_503_SERVICE_UNAVAILABLE,
    status.HTTP_504_GATEWAY_TIMEOUT,
}

@app.post(
    "/v1/chat/completions",
    # response_model=ForwarderResponse, # Return backend response or HTTPException
    summary="Forward Chat Completion Request",
    description=(
        "Accepts an OpenAI-compatible chat completion request. "
        "If 'model' is specified, it forwards to the corresponding provider. "
        "If 'model' is omitted, it attempts to use providers defined in `PROVIDERS_PRIORITY` "
        "with their configured default models, falling back on error."
    ),
    tags=["Forwarding"]
)
async def forward_chat_completion(
    request: ChatCompletionRequest,
    forwarder_key: str = Depends(security.validate_forwarder_key) # Authenticate the user of *this* service
):
    if not lifespan_client:
            logger.error("HTTP client not initialized during request.")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service unavailable: HTTP client not initialized.")

    last_error_response: Optional[ForwarderResponse] = None # Store last error for final exception

    # --- Scenario 1: Model is explicitly provided ---
    if request.model:
        logger.info(f"Received authenticated request for specific model: {request.model} (Key: {forwarder_key[:10]}...)")
        forwarded_response = await forward_request_to_provider(request, lifespan_client)

        if forwarded_response.success:
            logger.info(f"Forwarding successful for model {request.model}. Returning backend response.")
            return JSONResponse(content=forwarded_response.data)
        else:
            # Raise exception based on the single attempt's error
            error_status_code = forwarded_response.status_code or status.HTTP_502_BAD_GATEWAY
            error_detail = {
                "error_type": "forwarding_error",
                "message": forwarded_response.error or "Failed to forward request to backend provider.",
                "provider": forwarded_response.provider,
                "model_used": forwarded_response.model_used,
                "provider_error_details": forwarded_response.error_details
            }
            logger.error(f"Forwarding failed for explicit model {request.model}: Status={error_status_code}, Details={error_detail}")
            raise HTTPException(status_code=error_status_code, detail=error_detail)

    # --- Scenario 2: Model is NOT provided - Use Fallback Logic ---
    else:
        logger.info(f"Received authenticated request without specific model. Attempting fallback. (Key: {forwarder_key[:10]}...)")
        priority_list = config.settings.provider_priority_list
        default_models = config.settings.default_models

        if not priority_list:
            logger.error("Fallback failed: PROVIDERS_PRIORITY list is empty or not configured.")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request must include a 'model' or the server must have PROVIDERS_PRIORITY configured for fallback."
            )

        logger.info(f"Attempting fallback using priority: {priority_list}")

        for provider_name in priority_list:
            default_model = default_models.get(provider_name)
            if not default_model:
                logger.warning(f"Fallback: Skipping provider '{provider_name}' - No default model ({provider_name.upper()}_MODEL) configured.")
                continue

            # Create a *copy* of the request and set the model for this attempt
            request_copy = request.model_copy(deep=True) # Use model_copy for Pydantic v2
            request_copy.model = default_model
            
            logger.info(f"Fallback: Attempting provider '{provider_name}' with model '{default_model}'...")

            forwarded_response = await forward_request_to_provider(request_copy, lifespan_client)
            last_error_response = forwarded_response # Store the latest response (error or success)

            if forwarded_response.success:
                logger.info(f"Fallback successful with provider '{provider_name}' (model: {default_model}). Returning backend response.")
                # Return the successful response immediately
                return JSONResponse(content=forwarded_response.data)
            else:
                # Log the error for this provider attempt
                error_status = forwarded_response.status_code or 'N/A'
                logger.warning(f"Fallback: Attempt failed for provider '{provider_name}' (model: {default_model}). Status: {error_status}, Error: {forwarded_response.error}")

                # Decide if we should continue to the next provider
                is_retryable = (
                    forwarded_response.status_code in RETRYABLE_STATUS_CODES
                    or "Network error" in (forwarded_response.error or "") # Include explicit network errors
                    or forwarded_response.status_code is None # Treat unknown errors as potentially retryable
                )
                # Also retry if the error is 'API key not configured' for *this specific provider*
                is_config_error = "API key" in (forwarded_response.error or "") and "not configured" in (forwarded_response.error or "")

                if is_retryable or is_config_error:
                    logger.info(f"Fallback: Error for '{provider_name}' is considered retryable. Trying next provider.")
                    continue # Try next provider in the list
                else:
                    # Non-retryable error for this provider (e.g., 400 Bad Request specific to this provider's API, 401/403 auth error)
                    # We *could* stop here, but the current logic tries all providers.
                    # Let's log it and continue, the final error will be raised if all fail.
                    logger.warning(f"Fallback: Error for '{provider_name}' (Status: {error_status}) is considered non-retryable, but continuing fallback loop.")
                    continue # Continue for now, might change this behaviour later

        # --- End of Fallback Loop: All providers failed ---
        logger.error(f"Fallback failed: All providers in the priority list ({priority_list}) failed.")
        if last_error_response:
            # Raise an exception based on the *last* error encountered
            error_status_code = last_error_response.status_code or status.HTTP_502_BAD_GATEWAY
            error_detail = {
                "error_type": "fallback_error",
                "message": f"All fallback providers failed. Last error from '{last_error_response.provider}': {last_error_response.error}",
                "last_provider_attempted": last_error_response.provider,
                "last_model_attempted": last_error_response.model_used,
                "last_provider_status": last_error_response.status_code,
                "last_provider_error_details": last_error_response.error_details
            }
            logger.error(f"Raising final fallback error: Status={error_status_code}, Details={error_detail}")
            raise HTTPException(status_code=error_status_code, detail=error_detail)
        else:
            # Should not happen if priority_list was not empty, but as a safeguard
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Fallback failed, and no specific error information was recorded."
            )


# Add more endpoints if needed (e.g., /v1/models to list available models)

# --- Typer CLI App ---
cli_app = typer.Typer()

@cli_app.command()
def generate_key(length: int = typer.Option(32, help="Length of the random part of the key.")):
    """Generates a new secure API key and saves it to the keys file."""
    new_key = security.generate_api_key(length)
    keys_path = Path(__file__).parent.parent / "keys" / "forwarder_api_keys"
    keys_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure keys/ exists
    try:
        with keys_path.open("a") as f:
            # Add newline before appending if file is not empty
            if keys_path.stat().st_size > 0:
                 f.write("\n")
            f.write(new_key)
        print(f"\n--- Generated API Key ---\n")
        print(f"Generated Forwarder API Key: {new_key}")
        print(f"Appended to: {keys_path}\n")
        print(f"Make sure the key file does not contain empty lines or extra whitespace.")
        print(f"-" * 25 + "\n")
    except Exception as e:
         print(f"\n--- Error Saving Key ---")
         print(f"Could not write key to {keys_path}: {e}")
         print(f"Generated Key (Manually add if needed): {new_key}")
         print(f"-" * 25 + "\n")


@cli_app.command()
def run_server(
    host: str = typer.Option("127.0.0.1", help="Host to bind the server to."),
    port: int = typer.Option(8000, help="Port to run the server on."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reloading for development."),
    log_level: str = typer.Option("info", help="Log level (e.g., debug, info, warning, error).")
):
    """Runs the FastAPI web server."""
    # Ensure settings are loaded and processed before printing status
    # Accessing settings triggers the loading and validation logic
    s = config.settings
    print(f"--- AI Forwarder Server (v{app.version}) ---") # Show version
    print(f"Starting server on http://{host}:{port}")
    print(f"API Docs available at http://{host}:{port}/docs")
    print(f"Log Level: {log_level.upper()}")
    print(f"Auto-reload: {'Enabled' if reload else 'Disabled'}")
    print(f"Allowed Forwarder Keys Loaded: {len(s.allowed_forwarder_keys)}")
    if 0 < len(s.allowed_forwarder_keys) < 5:
         print(f"  Keys: {', '.join([k[:10]+'...' for k in s.allowed_forwarder_keys])}") # Show partial keys if few

    print(f"Provider Fallback Priority: {s.provider_priority_list or 'Not Configured'}")

    print(f"Backend Provider Status:")
    provider_status = _get_provider_status()
    configured_providers = 0
    for provider, status_info in provider_status.items():
            print(f"  - {provider.capitalize():<10}: {'Configured' if status_info['configured'] else 'Not Configured'} (Default Model: {status_info['default_model']})")
            if status_info['configured']: configured_providers += 1

    print("-" * (30 + max(len(p) for p in provider_status.keys()))) # Dynamic separator width

    # Warnings based on loaded settings
    if not s.allowed_forwarder_keys:
        print("\nWARNING: No valid FORWARDER_API_KEYS loaded from 'keys/forwarder_api_keys'.")
        print("         Run 'python -m app.main generate-key' or create the file manually.")
    elif s.allowed_forwarder_keys == ["fwd_key_replace_me"]:
         print("\nWARNING: Default placeholder 'fwd_key_replace_me' detected. Replace with generated keys.")

    if configured_providers == 0:
            print("\nWARNING: No backend provider API keys seem to be configured in .env.")
            print("         Forwarding will likely fail. Add relevant keys to your .env file.")
    elif not s.provider_priority_list and configured_providers > 0:
         print("\nINFO:    No PROVIDERS_PRIORITY set in .env. Fallback for requests without a model is disabled.")
         print("         Requests MUST include a 'model' field.")
    elif s.provider_priority_list:
         missing_defaults = [p for p in s.provider_priority_list if not s.default_models.get(p)]
         if missing_defaults:
             print(f"\nWARNING: Providers in priority list lack default models: {missing_defaults}")
             print(f"         Fallback will skip these. Set e.g., {missing_defaults[0].upper()}_MODEL in .env.")


    print("-" * (30 + max(len(p) for p in provider_status.keys())))


    uvicorn.run(
        "app.main:app", # Point to the FastAPI app instance
        host=host,
        port=port,
        reload=reload,
        log_level=log_level.lower(),
        lifespan="on" # Ensure lifespan events are handled
    )

# Optional: Add a CLI command to *use* the forwarder for quick testing
@cli_app.command()
def call(
    prompt: str = typer.Argument(..., help="The user prompt."),
    # Make model truly optional for testing fallback
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name (e.g., gpt-4o-mini, claude-3-haiku, groq:gemma2-9b-it). If omitted, tests server fallback."),
    key: str = typer.Option(None, "--key", "-k", help="Your Forwarder API Key (starts with fwd_). Get from keys file or generate-key. Can also use FORWARDER_KEY env var.", envvar="FORWARDER_KEY"), # Allow reading from env var
    server: str = typer.Option("http://127.0.0.1:8000", help="URL of the running forwarder server."),
    temperature: Optional[float] = typer.Option(None, "--temp", "-t", help="Temperature for sampling."),
    max_tokens: Optional[int] = typer.Option(None, "--max-tok", help="Max tokens to generate."), # Shortened option
    system_prompt: Optional[str] = typer.Option(None, "--system", "-s", help="Optional system prompt."),
):
    """Sends a chat request via the AI Forwarder (for testing)."""
    if not key:
        # Try reading from settings if env var was set but not passed via CLI
        loaded_key = os.getenv("FORWARDER_KEY")
        if not loaded_key:
             print("Error: Forwarder API Key is required. Use --key or set the FORWARDER_KEY environment variable.")
             raise typer.Exit(code=1)
        key = loaded_key
        print("Using FORWARDER_KEY from environment.")


    if not key.startswith(security.FORWARDER_KEY_PREFIX):
            print(f"Warning: Key doesn't start with '{security.FORWARDER_KEY_PREFIX}'. Make sure it's a forwarder key for this service.")

    api_endpoint = f"{server.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = { # Ensure payload is explicitly typed
        "messages": messages,
    }
    # Add optional parameters only if they are provided
    if model: # Only include model if provided
        payload["model"] = model
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
            payload["max_tokens"] = max_tokens
    # We generally don't send stream=True from basic CLI test
    # payload["stream"] = False

    print(f"--- Sending Request ---")
    print(f"Target Server: {server}")
    print(f"API Endpoint: {api_endpoint}")
    if model:
        print(f"Model Specified: {model}")
    else:
        print(f"Model Specified: None (Testing Server Fallback)")
    print(f"Payload Preview: {json.dumps(payload, indent=2)}")
    print(f"-----------------------")


    try:
        # Use httpx for consistency
        # Use a longer timeout matching the server's lifespan client
        timeout = httpx.Timeout(10.0, read=130.0) # Slightly longer than server read timeout
        with httpx.Client(timeout=timeout) as client:
            response = client.post(api_endpoint, headers=headers, json=payload)

            print(f"\n--- Response (Status: {response.status_code}) ---")

            try:
                # Try pretty printing if JSON, works for success and FastAPI JSON errors
                response_json = response.json()
                print(json.dumps(response_json, indent=2))

                # If it was a fallback error, print the specific provider details nicely
                if response.status_code >= 400 and isinstance(response_json.get("detail"), dict):
                     detail = response_json["detail"]
                     if detail.get("error_type") == "fallback_error":
                         print("\n--- Fallback Error Summary ---")
                         print(f"Message: {detail.get('message')}")
                         print(f"Last Provider Attempted: {detail.get('last_provider_attempted')}")
                         print(f"Last Model Attempted: {detail.get('last_model_attempted')}")
                         print(f"Last Provider Status: {detail.get('last_provider_status')}")
                         print("----------------------------")

            except (json.JSONDecodeError, AttributeError):
                    print(response.text) # Print raw text if not JSON

    except httpx.HTTPStatusError as e:
            # This might be redundant if FastAPI already formatted the error, but good fallback
            print(f"\n--- HTTP Error ({e.response.status_code}) ---")
            try:
                print(json.dumps(e.response.json(), indent=2))
            except (json.JSONDecodeError, AttributeError):
                print(e.response.text)
    except httpx.ReadTimeout:
         print(f"\n--- Request Error ---")
         print(f"The request timed out while waiting for a response from {server}.")
         print("The backend provider might be taking too long to generate the completion.")
         print("Consider increasing timeouts or checking the provider status.")
    except httpx.RequestError as e:
            print(f"\n--- Request Error ---")
            print(f"Could not connect to the forwarder server at {server}. Is it running?")
            print(f"Error details: {type(e).__name__} - {e}")
            print(f"Check the host/port and ensure the server is started with 'python -m app.main run-server'")
    except Exception as e:
            print(f"\n--- Unexpected CLI Error ---")
            print(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    # This allows running CLI commands like: python -m app.main generate-key
    # It loads settings implicitly when commands like run_server or call are invoked.
    cli_app()