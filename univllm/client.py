"""Universal LLM client with factory pattern for provider selection."""

from typing import AsyncIterator, Dict, Optional

from .exceptions import ConfigurationError, ModelNotSupportedError, ProviderError
from .models import (
    CompletionRequest,
    CompletionResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    Message,
    MessageRole,
    ModelCapabilities,
    ProviderType,
    ToolDefinition,
)
from .providers import (
    AnthropicProvider,
    BaseLLMProvider,
    DeepseekProvider,
    GeminiProvider,
    MistralProvider,
    OpenAIProvider,
)


class UniversalLLMClient:
    """Universal client for interacting with different LLM providers."""

    _provider_classes = {
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.ANTHROPIC: AnthropicProvider,
        ProviderType.DEEPSEEK: DeepseekProvider,
        ProviderType.MISTRAL: MistralProvider,
        ProviderType.GEMINI: GeminiProvider,
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
        self.config = kwargs

    def _resolve_provider(
        self, model: str, provider: Optional[ProviderType] = None
    ) -> ProviderType:
        """Resolve the provider for a request."""
        return provider or self._auto_detect_provider(model)

    def _ensure_provider(self, provider: ProviderType) -> BaseLLMProvider:
        """Return the active provider instance, rejecting implicit provider switches."""
        if not self.provider_instance:
            self._initialize_provider(provider, **self.config)
            return self.provider_instance

        if self.provider_type != provider:
            raise ConfigurationError(
                "This client is already initialized for "
                f"{self.provider_type.value}. Use set_provider() or create a new "
                f"client for {provider.value}."
            )

        return self.provider_instance

    @staticmethod
    def _normalize_messages(messages: list) -> list[Message]:
        """Normalize user-supplied messages to Message models."""
        processed_messages = []

        for msg in messages:
            if isinstance(msg, str):
                raise ValueError(
                    "messages must be Message objects or {'role', 'content'} dicts; "
                    "raw strings are not supported."
                )

            if isinstance(msg, Message):
                processed_messages.append(msg)
                continue

            if isinstance(msg, dict):
                try:
                    processed_messages.append(
                        Message(role=MessageRole(msg["role"]), content=msg["content"])
                    )
                except KeyError as exc:
                    raise ValueError(
                        "message dictionaries must include 'role' and 'content'"
                    ) from exc
                continue

            raise TypeError(
                "messages must contain only Message objects or {'role', 'content'} dicts"
            )

        return processed_messages

    @staticmethod
    def _normalize_tools(tools: Optional[list]) -> Optional[list[ToolDefinition]]:
        """Normalize user-supplied tools to ToolDefinition models."""
        if not tools:
            return None

        processed_tools = []
        for tool in tools:
            if isinstance(tool, ToolDefinition):
                processed_tools.append(tool)
            elif isinstance(tool, dict):
                processed_tools.append(ToolDefinition(**tool))
            else:
                raise TypeError(
                    "tools must contain only ToolDefinition objects or MCP-compatible dicts"
                )

        return processed_tools

    def _auto_detect_provider(self, model: str) -> ProviderType:
        """Auto-detect provider based on model name without instantiation."""
        for provider_type, provider_class in self._provider_classes.items():
            try:
                if provider_class.supports_model(model):  # type: ignore[attr-defined]
                    return provider_type
            except Exception:
                continue

        model_lower = model.lower()

        if any(keyword in model_lower for keyword in ["gpt", "openai"]):
            return ProviderType.OPENAI
        if any(keyword in model_lower for keyword in ["claude", "anthropic"]):
            return ProviderType.ANTHROPIC
        if "deepseek" in model_lower:
            return ProviderType.DEEPSEEK
        if any(keyword in model_lower for keyword in ["mistral", "mixtral"]):
            return ProviderType.MISTRAL
        if "gemini" in model_lower:
            return ProviderType.GEMINI

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

    @classmethod
    def get_supported_models(
        cls, provider: Optional[ProviderType] = None
    ) -> Dict[ProviderType, list]:
        """Get supported models for all or a specific provider."""
        if provider:
            if provider not in cls._provider_classes:
                raise ProviderError(f"Unsupported provider: {provider}")
            provider_class = cls._provider_classes[provider]
            return {provider: list(getattr(provider_class, "SUPPORTED_MODELS", []))}

        all_models: Dict[ProviderType, list] = {}
        for provider_type, provider_class in cls._provider_classes.items():
            all_models[provider_type] = list(
                getattr(provider_class, "SUPPORTED_MODELS", [])
            )

        return all_models

    def get_model_capabilities(
        self, model: str, provider: Optional[ProviderType] = None
    ) -> ModelCapabilities:
        """Get capabilities for a specific model."""
        resolved_provider = self._resolve_provider(model, provider)
        if resolved_provider not in self._provider_classes:
            raise ProviderError(f"Unsupported provider: {resolved_provider}")

        provider_class = self._provider_classes[resolved_provider]
        return provider_class.get_model_capabilities(model)

    async def complete(
        self,
        messages: list,
        model: str,
        provider: Optional[ProviderType] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
        **kwargs,
    ) -> CompletionResponse:
        """Generate a completion."""
        resolved_provider = self._resolve_provider(model, provider)
        processed_messages = self._normalize_messages(messages)
        processed_tools = self._normalize_tools(tools)
        provider_instance = self._ensure_provider(resolved_provider)

        request = CompletionRequest(
            messages=processed_messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            tools=processed_tools,
            tool_choice=tool_choice,
            extra_params=kwargs,
        )

        return await provider_instance.complete(request)

    async def stream_complete(
        self,
        messages: list,
        model: str,
        provider: Optional[ProviderType] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Generate a streaming completion."""
        resolved_provider = self._resolve_provider(model, provider)
        processed_messages = self._normalize_messages(messages)
        processed_tools = self._normalize_tools(tools)
        provider_instance = self._ensure_provider(resolved_provider)

        request = CompletionRequest(
            messages=processed_messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            tools=processed_tools,
            tool_choice=tool_choice,
            extra_params=kwargs,
        )

        async for chunk in provider_instance.stream_complete(request):
            yield chunk

    async def generate_image(
        self,
        prompt: str,
        model: str,
        provider: Optional[ProviderType] = None,
        size: Optional[str] = None,
        response_format: str = "b64_json",
        **kwargs,
    ) -> ImageGenerationResponse:
        """Generate an image using a vision/image capable model."""
        resolved_provider = self._resolve_provider(model, provider)
        provider_instance = self._ensure_provider(resolved_provider)
        request = ImageGenerationRequest(
            prompt=prompt,
            model=model,
            size=size,
            response_format=response_format,
            extra_params=kwargs,
        )
        return await provider_instance.generate_image(request)
