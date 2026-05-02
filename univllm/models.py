"""Data models for LLM interactions."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProviderType(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    GEMINI = "gemini"


class MessageRole(str, Enum):
    """Message roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """A message in a conversation."""

    role: MessageRole
    content: str


class ToolDefinition(BaseModel):
    """Definition of a tool/function that can be called by the LLM.
    
    This follows the MCP (Model Context Protocol) format for tool definitions.
    """

    name: str
    description: str
    input_schema: Dict[str, Any] = Field(
        description="JSON Schema defining the tool's input parameters"
    )


class ToolCall(BaseModel):
    """Represents a tool call requested by the LLM."""

    id: Optional[str] = None
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result from executing a tool call."""

    tool_call_id: Optional[str] = None
    content: str
    is_error: bool = False


class ModelCapabilities(BaseModel):
    """Capabilities of a specific model."""

    supports_system_messages: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None

AcceptedModel = str


class CompletionRequest(BaseModel):
    """Request for text completion."""

    messages: List[Message]
    model: AcceptedModel
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[str] = None  # "auto", "none", or specific tool name
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class CompletionResponse(BaseModel):
    """Response from text completion."""

    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    provider: ProviderType
    tool_calls: Optional[List[ToolCall]] = None


# --- Image generation models ---
class ImageGenerationRequest(BaseModel):
    """Request for image generation.

    size is optional; provider will apply a model-specific default (e.g. 'auto' for gpt-image-1).
    """

    prompt: str
    model: AcceptedModel
    size: Optional[str] = None
    response_format: str = "b64_json"  # or 'url'
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class GeneratedImage(BaseModel):
    """Represents a single generated image payload."""

    b64_json: Optional[str] = None
    url: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    """Response containing generated images."""

    images: List[GeneratedImage]
    model: str
    provider: ProviderType
    created: Optional[int] = None
    prompt: str
