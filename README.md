# 🚀 AI Forwarder <img src="static/logo.png" alt="Logo" width="40" style="vertical-align: middle; margin-left: 10px;">

A versatile FastAPI-based service that securely forwards API requests to various AI providers. It acts as a unified gateway, managing backend API keys and offering features like provider fallback.

## ✨ Features

- **Unified API Endpoint**: Single endpoint (`/v1/chat/completions`) for interacting with multiple AI backends.
- **Secure Backend Key Management**: Your precious OpenAI, Anthropic, Google, etc., API keys are stored securely on the server (via `.env`) and never exposed to the client.
- **Forwarder Authentication**: Protects the forwarder service itself using its own set of API keys.
- **Model Routing**: Intelligently routes requests to the correct provider based on model name prefixes (e.g., `gpt-`, `claude-`, `gemini-`, `groq:`, `openrouter:`, `ollama:`).
- **Provider Fallback**: Configure a priority list of providers and default models to use if a client sends a request without specifying a model, or if a primary provider fails (for retryable errors).
- **OpenAI-Compatible Interface**: Accepts requests in the standard OpenAI chat completions format.
- **Easy Configuration**: Uses a `.env` file for simple setup.
- **CLI Tools**: Includes commands to generate forwarder keys, run the server, and make test calls.
- **Interactive API Documentation**: Swagger UI and ReDoc available out-of-the-box.
- **Async Architecture**: Built with FastAPI and HTTPX for non-blocking I/O.
- **Supported Providers**:
  - OpenAI (e.g., `gpt-4o-mini`)
  - Anthropic (e.g., `claude-3-haiku-20240307`)
  - Google Gemini (e.g., `gemini-1.5-flash`)
  - Groq (e.g., `groq:gemma2-9b-it`)
  - OpenRouter (e.g., `openrouter:openai/gpt-4o-mini`)
  - Ollama (e.g., `ollama:llama3`)

## 🛠️ Prerequisites

- Python 3.8+
- `venv` (recommended for virtual environment management)

## 🚀 Getting Started

1.  **Clone the Repository & Navigate to Project Directory**

2.  **Create and Activate a Python Virtual Environment**

    - Windows PowerShell:
      ```bash
      python -m venv venv
      .\venv\Scripts\Activate.ps1
      ```
    - Linux/macOS/Git Bash:
      ```bash
      python -m venv venv
      source venv/bin/activate
      ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables for Backend Providers**

    Copy the example environment file:

    ```bash
    cp .env.example .env
    ```

    Now, edit the newly created `.env` file and **add your actual API keys** for the AI providers you want to use (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
    See the "⚙️ Configuration Deep Dive" section below for more details on `.env` options.

5.  **Generate Forwarder API Keys**

    These keys are used to authenticate requests _to this forwarder service_.
    Run the following command:

    ```bash
    python -m app.main generate-key
    ```

    This will:

    - Generate a new secure API key (e.g., `fwd_xxxxxxxx`).
    - Automatically create a `keys/` directory (if it doesn't exist).
    - Append the new key to `keys/forwarder_api_keys.txt`.
      The server will load allowed keys from this file. You can run this command multiple times to generate multiple keys.

6.  **Run the Server**

    ```bash
    python -m app.main run-server
    ```

    For development, enable auto-reloading:

    ```bash
    python -m app.main run-server --reload
    ```

## 🌐 Accessing the Application

Once the server is running (defaults to `http://127.0.0.1:8000`):

- **Status & Info Page**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive API Documentation (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative API Documentation (ReDoc)**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## ⚙️ Configuration Deep Dive (`.env` file)

The `.env` file is crucial for configuring the AI Forwarder.

### Backend API Keys (Required for forwarding)

Replace placeholders with your actual API keys.

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxx
GOOGLE_API_KEY=AIxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
# OLLAMA_API_KEY=ollama_key_if_needed # Optional, only if your Ollama instance requires auth
```

### Backend Base URLs (Optional)

Defaults are usually fine. Uncomment and change if you use a proxy or custom endpoint.

```env
# OPENAI_BASE_URL=https://api.openai.com/v1
# ANTHROPIC_BASE_URL=https://api.anthropic.com/v1
# ... and so on for other providers
# OLLAMA_BASE_URL=http://localhost:11434 # Default for local Ollama
```

### Provider Fallback Configuration (Optional)

Used when a request is sent to `/v1/chat/completions` _without_ a `model` specified.

- `PROVIDERS_PRIORITY`: Comma-separated list of provider names (e.g., `groq,openai,anthropic`). The forwarder will try them in this order.
  - Names must match keys in `PROVIDER_MAPPING` (lowercase: `openai`, `anthropic`, `google`, `groq`, `openrouter`, `ollama`).
  ```env
  # Example:
  # PROVIDERS_PRIORITY=groq,openai,anthropic
  ```
- Default models for each provider in the fallback list. The model format should be what you'd normally send.
  ```env
  # Example:
  # OPENAI_MODEL=gpt-4o-mini
  # ANTHROPIC_MODEL=claude-3-haiku-20240307
  # GOOGLE_MODEL=gemini-1.5-flash
  # GROQ_MODEL=groq:gemma2-9b-it        # Note: Groq models often need the "groq:" prefix here
  # OPENROUTER_MODEL=openrouter:openai/gpt-4o-mini # OpenRouter models need "openrouter:" prefix
  # OLLAMA_MODEL=ollama:llama3           # Ollama models need "ollama:" prefix
  ```

### Forwarder Security (CLI Convenience)

The `FORWARDER_KEY` variable in `.env` is **only** used by the `python -m app.main call ...` CLI command for convenience.
The server secures its endpoints using keys from the `keys/forwarder_api_keys.txt` file.

```env
# FORWARDER_KEY=fwd_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx # One of the keys from keys/forwarder_api_keys.txt
```

## 💻 CLI Commands

Use these commands from the project's root directory (where `requirements.txt` is).

1.  **`generate-key`**: Generates a new secure API key for accessing the forwarder.

    ```bash
    python -m app.main generate-key
    # Example with custom length:
    # python -m app.main generate-key --length 48
    ```

    This appends the key to `keys/forwarder_api_keys.txt`.

2.  **`run-server`**: Starts the FastAPI web server.

    ```bash
    python -m app.main run-server
    ```

    Options:

    - `--host TEXT`: Host to bind (default: `127.0.0.1`).
    - `--port INTEGER`: Port to use (default: `8000`).
    - `--reload`: Enable auto-reloading for development.
    - `--log-level TEXT`: Log level (e.g., `debug`, `info`, `warning`; default: `info`).

    ```bash
    # Example: Run on all interfaces, port 8080, with reload and debug logging
    # python -m app.main run-server --host 0.0.0.0 --port 8080 --reload --log-level debug
    ```

3.  **`call`**: Sends a test chat request to your running AI Forwarder.

    ```bash
    python -m app.main call "Tell me a joke about APIs."
    ```

    This command requires a Forwarder API key. It will try to use the `FORWARDER_KEY` environment variable (which can be set in `.env`) or you can provide it with `--key`.

    Options:

    - `PROMPT`: The user prompt (required).
    - `--model TEXT` / `-m TEXT`: Specific model to use (e.g., `gpt-4o-mini`, `claude-3-haiku-20240307`). If omitted, tests server fallback.
    - `--key TEXT` / `-k TEXT`: Your Forwarder API Key (starts with `fwd_`).
    - `--server TEXT`: URL of the running forwarder server (default: `http://127.0.0.1:8000`).
    - `--temp FLOAT` / `-t FLOAT`: Temperature for sampling.
    - `--max-tok INTEGER`: Max tokens to generate.
    - `--system TEXT` / `-s TEXT`: Optional system prompt.

    ```bash
    # Example: Specific model, key provided, system prompt
    # python -m app.main call --key fwd_yourkeyhere --model groq:llama3-8b-8192 --system "You are a pirate." "What's for dinner?"

    # Example: Testing fallback (model omitted), key from .env or FORWARDER_KEY env var
    # python -m app.main call "Summarize the theory of relativity in simple terms."
    ```

4.  **Show Help for All Commands**
    ```bash
    python -m app.main --help
    ```

## 🔌 API Usage

### Endpoint: `/v1/chat/completions`

- **Method**: `POST`
- **Authentication**: Requires an `Authorization` header.

  ```
  Authorization: Bearer YOUR_FORWARDER_API_KEY
  ```

  Replace `YOUR_FORWARDER_API_KEY` with a key from `keys/forwarder_api_keys.txt`.

- **Request Body**: OpenAI-compatible `ChatCompletionRequest` JSON.
  - `model` (String, Optional):
    - If provided, routes to the corresponding provider based on prefix.
      - `gpt-*` models (e.g., `gpt-4o-mini`, `gpt-3.5-turbo`) -> OpenAI
      - `claude-*` models (e.g., `claude-3-opus-20240229`, `claude-3-haiku-20240307`) -> Anthropic
      - `gemini-*` models (e.g., `gemini-1.5-pro-latest`, `gemini-1.5-flash`) -> Google
      - `groq:<model_name>` (e.g., `groq:llama3-8b-8192`, `groq:gemma2-9b-it`) -> Groq
      - `openrouter:<provider/model_name>` (e.g., `openrouter:openai/gpt-4o-mini`, `openrouter:anthropic/claude-3-haiku`) -> OpenRouter
      - `ollama:<model_name>` (e.g., `ollama:llama3`, `ollama:mistral`) -> Ollama
    - If omitted (or `null`), the server attempts fallback based on `PROVIDERS_PRIORITY` and `*_MODEL` settings in `.env`.
  - `messages` (Array of Objects, Required):
    - Each object: `{"role": "system" | "user" | "assistant", "content": "..."}`
  - `temperature` (Float, Optional)
  - `top_p` (Float, Optional)
  - `max_tokens` (Integer, Optional)
  - `stream` (Boolean, Optional, Default: `false`): **Note: Streaming is NOT fully supported by the forwarder logic yet.**
  - `stop` (String or Array of Strings, Optional)
  - `presence_penalty` (Float, Optional)
  - `frequency_penalty` (Float, Optional)
  - `user` (String, Optional)
  - `extra_body` (Object, Optional): For provider-specific parameters not part of the standard OpenAI API. These are passed through to the selected provider.

### Example `curl` Request

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_FORWARDER_API_KEY" \
-d '{
    "model": "groq:gemma2-9b-it",
    "messages": [
        {"role": "system", "content": "You are a witty and concise assistant."},
        {"role": "user", "content": "What is the capital of France and why is it famous?"}
    ],
    "temperature": 0.7,
    "max_tokens": 150
}'
```

### Example `curl` Request (Testing Fallback)

To test fallback, omit the `model` field (ensure `PROVIDERS_PRIORITY` and relevant `*_MODEL` vars are set in `.env`):

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_FORWARDER_API_KEY" \
-d '{
    "messages": [
        {"role": "user", "content": "Give me three fun facts about the ocean."}
    ],
    "temperature": 0.8
}'
```

## 📜 License

This project is licensed under the MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

---

Happy Forwarding! 🎉
