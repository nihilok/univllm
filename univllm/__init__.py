"""
Universal interface for different LLM providers.

This package provides a standardized way to interact with various LLM providers
including OpenAI, Anthropic, Deepseek, and Mistral.
"""

from .client import UniversalLLMClient
from .models import ProviderType
from .exceptions import UniversalLLMError, ProviderError, ModelNotSupportedError
from .supported_models import is_unsupported_model

__all__ = [
    "UniversalLLMClient",
    "ProviderType",
    "UniversalLLMError",
    "ProviderError",
    "ModelNotSupportedError",
    "is_unsupported_model",
]
