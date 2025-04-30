# app/main.py
import typer
import uvicorn
import httpx
import json
import logging
from typing import Optional
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
logger = logging.getLogger(__name__)

# --- Async Lifecycle for HTTPX client ---
# Create a global httpx client to reuse connections
lifespan_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global lifespan_client
    # Startup: Initialize the HTTPX client
    lifespan_client = httpx.AsyncClient()
    logger.info("HTTPX Client started.")
    yield
    # Shutdown: Close the HTTPX client
    if lifespan_client:
        await lifespan_client.aclose()
        logger.info("HTTPX Client closed.")

# --- FastAPI App Setup ---
app = FastAPI(
    title="AI Forwarder",
    description="Forwards requests to various AI providers using stored API keys.",
    version="0.2.0", # Updated version
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
         logger.warning(f"Static directory not found at: {static_dir}")

    if templates_dir.is_dir():
        templates = Jinja2Templates(directory=templates_dir)
        logger.info(f"Loaded HTML templates from: {templates_dir}")
    else:
        logger.warning(f"Templates directory not found at: {templates_dir}")

except Exception as e:
     logger.error(f"Error mounting static files or templates: {e}", exc_info=True)


# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root(request: Request):
    if not templates:
        return HTMLResponse("<html><body><h1>AI Forwarder</h1><p>Error: HTML Templates directory not found or not loaded. Check server logs.</p></body></html>", status_code=500)
    # Pass some info to the template
    provider_keys_status = {p: bool(k) for p, k in config.settings.backend_api_keys.items() if p in config.PROVIDER_MAPPING}
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "AI Forwarder Status",
        "forwarder_keys_count": len(config.settings.allowed_forwarder_keys),
        "provider_keys_status": provider_keys_status,
        "api_docs_url": app.docs_url or "/docs", # Provide defaults if None
        "redoc_url": app.redoc_url or "/redoc"
    })

@app.post(
    "/v1/chat/completions",
    # response_model=ForwarderResponse, # We return backend response directly on success, or HTTPException on error
    summary="Forward Chat Completion Request",
    description="Accepts an OpenAI-compatible chat completion request and forwards it to the appropriate backend AI provider based on the 'model' field.",
    tags=["Forwarding"]
)
async def forward_chat_completion(
    request: ChatCompletionRequest,
    forwarder_key: str = Depends(security.validate_forwarder_key) # Authenticate the user of *this* service
):
    if not lifespan_client:
            logger.error("HTTP client not initialized during request.")
            raise HTTPException(status_code=503, detail="Service unavailable: HTTP client not initialized.")

    logger.info(f"Received authenticated request for model: {request.model} (Key: {forwarder_key[:5]}...)") # Log model and partial key

    forwarded_response = await forward_request_to_provider(request, lifespan_client)

    if forwarded_response.success:
        # Return the data received from the backend directly
        # The response structure will vary by provider
        # Ensure data is serializable; usually it's a dict from response.json()
        logger.info(f"Forwarding successful for model {request.model}. Returning backend response.")
        return JSONResponse(content=forwarded_response.data)
    else:
        # Determine appropriate status code for the error
        error_status_code = status.HTTP_502_BAD_GATEWAY # Default for upstream errors
        if forwarded_response.status_code: # Use status code from backend if available
             error_status_code = forwarded_response.status_code
        elif "Unsupported model" in (forwarded_response.error or ""):
             error_status_code = status.HTTP_400_BAD_REQUEST
        elif "API key" in (forwarded_response.error or "") and "not configured" in (forwarded_response.error or ""):
             error_status_code = status.HTTP_503_SERVICE_UNAVAILABLE # Internal config issue
        elif "Network error" in (forwarded_response.error or ""):
             error_status_code = status.HTTP_504_GATEWAY_TIMEOUT # Suggests connection issue
        
        # Log the detailed error before raising HTTPException
        logger.error(f"Forwarding failed for model {request.model}: Status={error_status_code}, Error='{forwarded_response.error}', Details='{forwarded_response.error_details}'")

        # Raise HTTPException which FastAPI converts to a JSON error response
        raise HTTPException(
            status_code=error_status_code,
            detail={
                "error_type": "forwarding_error",
                "message": forwarded_response.error or "Failed to forward request to backend provider.",
                "provider": forwarded_response.provider,
                "model_used": forwarded_response.model_used,
                "provider_error_details": forwarded_response.error_details # Include backend error details if available
                }
        )

# Add more endpoints if needed (e.g., /v1/models to list available models)

# --- Typer CLI App ---
cli_app = typer.Typer()

@cli_app.command()
def generate_key(
    length: int = typer.Option(32, help="Length of the random part of the key.")
):
    """Generates a new secure API key for accessing the forwarder service."""
    new_key = security.generate_api_key(length)
    print(f"Generated Forwarder API Key: {new_key}")
    print("\nAdd this key to the FORWARDER_API_KEYS variable in your .env file (comma-separated).")
    print(f"Example: FORWARDER_API_KEYS=\"existing_key1,{new_key}\"")

@cli_app.command()
def run_server(
    host: str = typer.Option("127.0.0.1", help="Host to bind the server to."),
    port: int = typer.Option(8000, help="Port to run the server on."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reloading for development."),
    log_level: str = typer.Option("info", help="Log level (e.g., debug, info, warning, error).")
):
    """Runs the FastAPI web server."""
    # Ensure settings are loaded and processed before printing status
    _ = config.settings 

    print(f"--- AI Forwarder Server ---")
    print(f"Starting server on http://{host}:{port}")
    print(f"API Docs available at http://{host}:{port}/docs")
    print(f"Log Level: {log_level}")
    print(f"Auto-reload: {'Enabled' if reload else 'Disabled'}")
    print(f"Allowed Forwarder Keys Loaded: {len(config.settings.allowed_forwarder_keys)}")
    if len(config.settings.allowed_forwarder_keys) > 0 and len(config.settings.allowed_forwarder_keys) < 5:
         print(f"  Keys: {', '.join([k[:5]+'...' for k in config.settings.allowed_forwarder_keys])}") # Show partial keys if few

    print(f"Backend Provider Status:")
    configured_providers = 0
    for provider, key in config.settings.backend_api_keys.items():
            if provider in config.PROVIDER_MAPPING: # Only show relevant providers
                is_configured = bool(key)
                print(f"  - {provider.capitalize():<10}: {'Loaded' if is_configured else 'Not Configured'}")
                if is_configured: configured_providers += 1

    print("-" * 27)

    if not config.settings.allowed_forwarder_keys or config.settings.allowed_forwarder_keys == ["fwd_key_replace_me"]:
        print("\nWARNING: No valid FORWARDER_API_KEYS are configured in .env. The API will not be usable.")
        print("         Run 'python -m app.main generate-key' and add the key to .env.")
    if configured_providers == 0:
            print("\nWARNING: No backend provider API keys (e.g., OPENAI_API_KEY) are configured in .env.")
            print("         Forwarding will fail. Add relevant keys to your .env file.")
    print("-" * 27)


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
    model: str = typer.Option("gpt-4o-mini", help="Model name (e.g., gpt-4o-mini, claude-3-haiku-20240307)."),
    key: str = typer.Option(None, "--key", "-k", help="Your Forwarder API Key (starts with fwd_). Get from .env or generate-key. Can also use FORWARDER_KEY env var.", envvar="FORWARDER_KEY"), # Allow reading from env var
    server: str = typer.Option("http://127.0.0.1:8000", help="URL of the running forwarder server."),
    temperature: Optional[float] = typer.Option(None, "--temp", "-t", help="Temperature for sampling."),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", "-m", help="Max tokens to generate."),
    system_prompt: Optional[str] = typer.Option(None, "--system", "-s", help="Optional system prompt."),
):
    """Sends a simple chat request via the AI Forwarder (for testing)."""
    if not key:
        print("Error: Forwarder API Key is required. Use --key or set the FORWARDER_KEY environment variable.")
        raise typer.Exit(code=1)

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

    payload = {
        "model": model,
        "messages": messages,
    }
    # Add optional parameters only if they are provided
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
            payload["max_tokens"] = max_tokens

    print(f"--- Sending Request ---")
    print(f"Target Server: {server}")
    print(f"API Endpoint: {api_endpoint}")
    print(f"Model: {model}")
    print(f"Payload Preview: {json.dumps(payload, indent=2)}")
    print(f"-----------------------")


    try:
        # Use httpx for consistency and async context if needed later
        with httpx.Client(timeout=120.0) as client: # Match server timeout
            response = client.post(api_endpoint, headers=headers, json=payload)

            print(f"\n--- Response (Status: {response.status_code}) ---")

            try:
                # Try pretty printing if JSON, works for success and FastAPI JSON errors
                print(json.dumps(response.json(), indent=2))
            except json.JSONDecodeError:
                    print(response.text) # Print raw text if not JSON (e.g., 502 gateway errors)

            # Optionally raise status after printing for clarity in CLI tool
            # response.raise_for_status()

    except httpx.HTTPStatusError as e:
            # This might be redundant if FastAPI already formatted the error, but good fallback
            print(f"\n--- HTTP Error ({e.response.status_code}) ---")
            try:
                print(json.dumps(e.response.json(), indent=2))
            except json.JSONDecodeError:
                print(e.response.text)
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
