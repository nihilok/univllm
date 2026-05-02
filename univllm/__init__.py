"""
Universal interface for different LLM providers.

This package provides a standardized way to interact with various LLM providers
including OpenAI, Anthropic, Deepseek, and Mistral.
"""

from .client import UniversalLLMClient
from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    ModelNotSupportedError,
    ProviderError,
    UniversalLLMError,
)
from .models import (
    CompletionResponse,
    GeneratedImage,
    ImageGenerationResponse,
    Message,
    MessageRole,
    ModelCapabilities,
    ProviderType,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from .providers import BaseLLMProvider

__all__ = [
    "UniversalLLMClient",
    "ProviderType",
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "Message",
    "MessageRole",
    "ModelCapabilities",
    "CompletionResponse",
    "GeneratedImage",
    "ImageGenerationResponse",
    "BaseLLMProvider",
    "UniversalLLMError",
    "ProviderError",
    "ModelNotSupportedError",
    "AuthenticationError",
    "ConfigurationError",
]
