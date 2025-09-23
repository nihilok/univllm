"""Base provider class for LLM interactions."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, AsyncIterator
from ..models import (
    CompletionRequest,
    CompletionResponse,
    ModelCapabilities,
    ProviderType,
    ImageGenerationRequest,
    ImageGenerationResponse,
)


class BaseLLMProvider(ABC):
    """Base class for all LLM providers."""

    # Class-level list of supported model name prefixes (override in subclasses)
    SUPPORTED_MODELS: List[str] = []

    def __init__(self, api_key: Optional[str] = None, **kwargs) -> None:
        """Initialize the provider.

        Args:
            api_key: API key for the provider
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.config = kwargs

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        pass

    # Removed abstract requirement; keep instance property for backwards compatibility
    @property
    def supported_models(self) -> List[str]:
        """Return list of supported models (class-level)."""
        return type(self).SUPPORTED_MODELS

    @classmethod
    def supports_model(cls, model: str) -> bool:
        """Class-level check if model is supported without instantiation."""
        return any(model.startswith(supported) for supported in cls.SUPPORTED_MODELS)

    @abstractmethod
    def get_model_capabilities(self, model: str) -> ModelCapabilities:
        """Get capabilities for a specific model.

        Args:
            model: Model identifier

        Returns:
            ModelCapabilities object
        """
        pass

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion for the given request.

        Args:
            request: Completion request

        Returns:
            Completion response
        """
        pass

    @abstractmethod
    async def stream_complete(self, request: CompletionRequest) -> AsyncIterator[str]:
        """Generate a streaming completion for the given request.

        Args:
            request: Completion request

        Yields:
            Chunks of the completion
        """
        pass

    # --- Optional image generation interface ---
    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResponse:  # pragma: no cover - default impl
        """Generate images for the given request.

        Providers that support image generation should override this.
        """
        raise NotImplementedError(
            f"Image generation not implemented for provider {self.provider_type}"
        )

    def validate_model(self, model: str) -> bool:
        """Validate if a model is supported by this provider."""
        return type(self).supports_model(model)

    def prepare_request(self, request: CompletionRequest) -> Dict:
        """Prepare request data for the provider's API format.

        Args:
            request: Completion request

        Returns:
            Dictionary formatted for the provider's API
        """
        # Base implementation - providers can override
        messages = [
            {"role": msg.role.value, "content": msg.content} for msg in request.messages
        ]

        data = {
            "model": request.model,
            "messages": messages,
        }

        if request.max_tokens is not None:
            data["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            data["temperature"] = request.temperature
        if request.top_p is not None:
            data["top_p"] = request.top_p
        if request.stream:
            data["stream"] = request.stream

        # Add any extra parameters
        data.update(request.extra_params)

        return data
