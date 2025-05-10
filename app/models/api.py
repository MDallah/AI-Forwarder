# app/models/api.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

class ChatMessage(BaseModel):
    role: str # "system", "user", "assistant"
    content: Optional[str] = None # Content is optional (e.g., function calls)

class ChatCompletionRequest(BaseModel):
    # Make model optional - will trigger fallback if None/empty
    model: Optional[str] = Field(None, description="The model to use for the chat completion. If omitted, the server will attempt fallback based on PROVIDERS_PRIORITY.")
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = Field(False, description="Whether to stream the response. NOTE: Streaming is NOT fully supported by the forwarder logic yet.")
    # Add other common OpenAI parameters as Optional fields
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None
    # Allow arbitrary provider-specific parameters
    extra_body: Optional[Dict[str, Any]] = Field(None, description="Provider-specific parameters not part of the standard OpenAI API.")

    # Example for validation if needed
    # @validator('temperature')
    # def temperature_range(cls, v):
    #     if v is not None and not (0 <= v <= 2):
    #         raise ValueError('temperature must be between 0 and 2')
    #     return v

class ForwarderResponse(BaseModel):
    """Standardized response wrapper for internal handling"""
    success: bool
    provider: Optional[str] = None
    model_used: Optional[str] = None
    data: Optional[Any] = None # Holds the successful response body from the backend
    error: Optional[str] = None # Holds a high-level error message
    status_code: Optional[int] = None # HTTP status code from the backend if an error occurred
    error_details: Optional[Any] = None # Holds detailed error info (e.g., parsed JSON error from backend)