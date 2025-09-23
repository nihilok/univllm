"""Data models for LLM interactions."""

from enum import Enum
from typing import Annotated, Any, Dict, List, Optional
from pydantic import AfterValidator, BaseModel, Field

from univllm.supported_models import is_potentially_supported_model


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


def validate_supported_model(model: str) -> str:
    """Validate that the model is potentially supported."""
    if not is_potentially_supported_model(model):
        raise ValueError(f"Model '{model}' is not supported")
    return model


AcceptedModel = Annotated[
    str,
    AfterValidator(validate_supported_model),
]


class CompletionRequest(BaseModel):
    """Request for text completion."""

    messages: List[Message]
    model: AcceptedModel
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


# --- Image generation models ---
class ImageGenerationRequest(BaseModel):
    """Request for image generation."""

    prompt: str
    model: AcceptedModel
    size: str = "512x512"
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
