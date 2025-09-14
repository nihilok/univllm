"""Data models for LLM interactions."""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ProviderType(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"


class MessageRole(str, Enum):
    """Message roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """A message in a conversation."""

    role: MessageRole
    content: str


class ModelCapabilities(BaseModel):
    """Capabilities of a specific model."""

    supports_system_messages: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None


class CompletionRequest(BaseModel):
    """Request for text completion."""

    messages: List[Message]
    model: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class CompletionResponse(BaseModel):
    """Response from text completion."""

    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    provider: ProviderType
