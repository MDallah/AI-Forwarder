# app/models/api.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# We'll try to mimic OpenAI's ChatCompletion API structure for simplicity
# Users can send requests in this format, and we'll adapt them if needed.

class ChatMessage(BaseModel):
    role: str # "system", "user", "assistant", "tool" (OpenAI specific)
    content: Optional[str] = None # Content can be None for some roles (e.g., tool calls)
    # Add tool_calls, tool_call_id etc. if supporting OpenAI functions/tools

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="The ID of the model to use (e.g., 'gpt-4o', 'claude-3-opus-20240229', 'gemini-1.5-flash').")
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False # NOTE: Streaming support is more complex, starting without it
    # Add other common parameters as needed (stop, presence_penalty, etc.)
    extra_body: Optional[Dict[str, Any]] = None # For provider-specific params

# A generic response structure, actual content depends on the backend
class ForwarderResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    provider: Optional[str] = None
    model_used: Optional[str] = None
    status_code: Optional[int] = None # Store status code from backend if error
    error_details: Optional[Any] = None # Store parsed error details from backend
