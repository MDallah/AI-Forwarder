# AI Forwarder

A service that forwards API requests to various AI providers.

### Getting Started

Follow these steps to set up and run your AI Forwarder:

1. **Navigate to the project directory**

   ```bash
   cd $ProjectName
   ```

2. **Create a Python virtual environment**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**

   Windows PowerShell:

   ```bash
   .\venv\Scripts\Activate.ps1
   ```

   Linux/macOS/Git Bash:

   ```bash
   source venv/bin/activate
   ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and add your real backend API keys (OpenAI, Anthropic, etc.)

6. **Generate forwarder keys**

   ```bash
   python -m app.main generate-key
   ```

   > Copy the generated key(s) into the `FORWARDER_API_KEYS` variable in your `.env` file

7. **Run the server**
   ```bash
   python -m app.main run-server --reload
   ```

### Access the Application

- **Status Page**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Features

- Forward API requests to multiple AI providers
- Secure API key management
- Simple configuration
- Interactive API documentation
