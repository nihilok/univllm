"""Universal LLM client with factory pattern for provider selection."""

from typing import Dict, Optional, AsyncIterator
from .models import (
    CompletionRequest,
    CompletionResponse,
    ModelCapabilities,
    Message,
    MessageRole,
    ProviderType,
    ImageGenerationRequest,
    ImageGenerationResponse,
)
from .providers import (
    BaseLLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    DeepseekProvider,
    MistralProvider,
)
from .exceptions import ProviderError, ModelNotSupportedError


class UniversalLLMClient:
    """Universal client for interacting with different LLM providers."""

    _provider_classes = {
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.ANTHROPIC: AnthropicProvider,
        ProviderType.DEEPSEEK: DeepseekProvider,
        ProviderType.MISTRAL: MistralProvider,
    }

    def __init__(self, provider: Optional[ProviderType] = None, **kwargs) -> None:
        """Initialize the universal client.

        Args:
            provider: The provider to use (if not specified, will auto-detect)
            **kwargs: Provider-specific configuration
        """
        self.provider_instance: Optional[BaseLLMProvider] = None
        self.provider_type = provider
        self.config = kwargs

        if provider:
            self._initialize_provider(provider, **kwargs)

    def _initialize_provider(self, provider: ProviderType, **kwargs) -> None:
        """Initialize a specific provider.

        Args:
            provider: The provider type to initialize
            **kwargs: Provider-specific configuration
        """
        if provider not in self._provider_classes:
            raise ProviderError(f"Unsupported provider: {provider}")

        provider_class = self._provider_classes[provider]
        self.provider_instance = provider_class(**kwargs)
        self.provider_type = provider

    def _auto_detect_provider(self, model: str) -> ProviderType:
        """Auto-detect provider based on model name without instantiation."""
        # First, try class-level supported model checks
        for provider_type, provider_class in self._provider_classes.items():
            try:
                if provider_class.supports_model(model):  # type: ignore[attr-defined]
                    return provider_type
            except Exception:
                continue

        # Fallback heuristics
        model_lower = model.lower()

        if any(keyword in model_lower for keyword in ["gpt", "openai"]):
            return ProviderType.OPENAI
        if any(keyword in model_lower for keyword in ["claude", "anthropic"]):
            return ProviderType.ANTHROPIC
        if "deepseek" in model_lower:
            return ProviderType.DEEPSEEK
        if any(keyword in model_lower for keyword in ["mistral", "mixtral"]):
            return ProviderType.MISTRAL

        raise ModelNotSupportedError(
            f"Could not auto-detect provider for model: {model}"
        )

    def set_provider(self, provider: ProviderType, **kwargs) -> None:
        """Set or change the provider.

        Args:
            provider: The provider type to use
            **kwargs: Provider-specific configuration
        """
        self._initialize_provider(provider, **kwargs)

    def get_supported_models(
        self, provider: Optional[ProviderType] = None
    ) -> Dict[ProviderType, list]:
        """Get supported models for all or specific provider using class-level data.

        Args:
            provider: Specific provider to get models for (if None, gets all)

        Returns:
            Dictionary mapping provider types to their supported models
        """
        if provider:
            if provider not in self._provider_classes:
                raise ProviderError(f"Unsupported provider: {provider}")
            provider_class = self._provider_classes[provider]
            return {provider: list(getattr(provider_class, "SUPPORTED_MODELS", []))}

        all_models: Dict[ProviderType, list] = {}
        for provider_type, provider_class in self._provider_classes.items():
            all_models[provider_type] = list(
                getattr(provider_class, "SUPPORTED_MODELS", [])
            )

        return all_models

    def get_model_capabilities(
        self, model: str, provider: Optional[ProviderType] = None
    ) -> ModelCapabilities:
        """Get capabilities for a specific model.

        Args:
            model: Model identifier
            provider: Provider to use (if not specified, will auto-detect)

        Returns:
            Model capabilities
        """
        if not provider:
            provider = self._auto_detect_provider(model)

        if not self.provider_instance or self.provider_type != provider:
            self._initialize_provider(provider, **self.config)

        return self.provider_instance.get_model_capabilities(model)

    async def complete(
        self,
        messages: list,
        model: str,
        provider: Optional[ProviderType] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> CompletionResponse:
        """Generate a completion.

        Args:
            messages: List of messages (can be strings or Message objects)
            model: Model identifier
            provider: Provider to use (if not specified, will auto-detect)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional provider-specific parameters

        Returns:
            Completion response
        """
        if not provider:
            provider = self._auto_detect_provider(model)

        if not self.provider_instance or self.provider_type != provider:
            self._initialize_provider(provider, **self.config)

        # Convert messages to Message objects if they're strings
        processed_messages = []
        for i, msg in enumerate(messages):
            if isinstance(msg, str):
                # First message is system if only one, otherwise alternate user/assistant
                if i == 0 and len(messages) == 1:
                    role = MessageRole.USER
                elif i == 0:
                    role = MessageRole.SYSTEM
                elif i % 2 == 1:
                    role = MessageRole.USER
                else:
                    role = MessageRole.ASSISTANT
                processed_messages.append(Message(role=role, content=msg))
            elif isinstance(msg, dict):
                processed_messages.append(
                    Message(role=MessageRole(msg["role"]), content=msg["content"])
                )
            else:
                processed_messages.append(msg)

        request = CompletionRequest(
            messages=processed_messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_params=kwargs,
        )

        return await self.provider_instance.complete(request)

    async def stream_complete(
        self,
        messages: list,
        model: str,
        provider: Optional[ProviderType] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Generate a streaming completion.

        Args:
            messages: List of messages (can be strings or Message objects)
            model: Model identifier
            provider: Provider to use (if not specified, will auto-detect)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional provider-specific parameters

        Yields:
            Chunks of the completion
        """
        if not provider:
            provider = self._auto_detect_provider(model)

        if not self.provider_instance or self.provider_type != provider:
            self._initialize_provider(provider, **self.config)

        # Convert messages to Message objects if they're strings
        processed_messages = []
        for i, msg in enumerate(messages):
            if isinstance(msg, str):
                # First message is system if only one, otherwise alternate user/assistant
                if i == 0 and len(messages) == 1:
                    role = MessageRole.USER
                elif i == 0:
                    role = MessageRole.SYSTEM
                elif i % 2 == 1:
                    role = MessageRole.USER
                else:
                    role = MessageRole.ASSISTANT
                processed_messages.append(Message(role=role, content=msg))
            elif isinstance(msg, dict):
                processed_messages.append(
                    Message(role=MessageRole(msg["role"]), content=msg["content"])
                )
            else:
                processed_messages.append(msg)

        request = CompletionRequest(
            messages=processed_messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            extra_params=kwargs,
        )

        async for chunk in self.provider_instance.stream_complete(request):
            yield chunk

    async def generate_image(
        self,
        prompt: str,
        model: str,
        provider: Optional[ProviderType] = None,
        size: str = "512x512",
        response_format: str = "b64_json",
        **kwargs,
    ) -> ImageGenerationResponse:
        """Generate an image using a vision/image capable model."""
        if not provider:
            provider = self._auto_detect_provider(model)
        if not self.provider_instance or self.provider_type != provider:
            self._initialize_provider(provider, **self.config)
        request = ImageGenerationRequest(
            prompt=prompt,
            model=model,
            size=size,
            response_format=response_format,
            extra_params=kwargs,
        )
        # Provider must implement generate_image; base raises NotImplementedError
        return await self.provider_instance.generate_image(request)
