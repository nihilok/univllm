"""Anthropic provider implementation."""

import os
from typing import List, Optional, AsyncIterator, Dict
import anthropic

from ..supported_models import ANTHROPIC_SUPPORTED_MODELS
from ..models import (
    CompletionRequest,
    CompletionResponse,
    ModelCapabilities,
    MessageRole,
    ProviderType,
)
from ..exceptions import ProviderError, ModelNotSupportedError, AuthenticationError
from .base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """Anthropic provider for Claude models."""

    SUPPORTED_MODELS: List[str] = ANTHROPIC_SUPPORTED_MODELS

    def __init__(self, api_key: Optional[str] = None, **kwargs) -> None:
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (if not provided, will use ANTHROPIC_API_KEY env var)
            **kwargs: Additional configuration
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise AuthenticationError("Anthropic API key is required")

        super().__init__(api_key=api_key, **kwargs)
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.ANTHROPIC

    def get_model_capabilities(self, model: str) -> ModelCapabilities:
        """Get capabilities for a specific Anthropic model."""
        if not self.validate_model(model):
            raise ModelNotSupportedError(
                f"Model {model} is not supported by Anthropic provider"
            )

        # Default capabilities for Anthropic models
        capabilities = ModelCapabilities(
            supports_system_messages=True,
            supports_function_calling=True,  # Claude models now support tool use
            supports_streaming=True,
            supports_vision=False,
        )

        # Model-specific capabilities based on latest Anthropic specifications
        if model.startswith("claude-3-7-sonnet"):
            # Claude 3.7 Sonnet - enhanced version
            capabilities.context_window = 200000
            capabilities.max_tokens = 8192
            capabilities.supports_vision = True
        elif model.startswith("claude-sonnet-4-"):
            # Claude Sonnet 4.x series - next generation
            capabilities.context_window = 200000
            capabilities.max_tokens = 8192
            capabilities.supports_vision = True
        elif model.startswith("claude-opus-4-1-"):
            # Claude Opus 4.1 series - most capable model
            capabilities.context_window = 200000
            capabilities.max_tokens = 8192
            capabilities.supports_vision = True
        elif model.startswith("claude-code"):
            # Claude Code - specialized for coding
            capabilities.context_window = 200000
            capabilities.max_tokens = 8192
            capabilities.supports_vision = False
            # Enhanced function calling for code generation
            capabilities.supports_function_calling = True

        return capabilities

    def prepare_request(self, request: CompletionRequest) -> Dict:
        """Prepare request data for Anthropic API format."""
        # Anthropic requires system messages to be separate
        system_message = None
        messages = []

        for msg in request.messages:
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content
            else:
                messages.append({"role": msg.role.value, "content": msg.content})

        data = {
            "model": request.model,
            "messages": messages,
        }

        if system_message:
            data["system"] = system_message

        if request.max_tokens is not None:
            data["max_tokens"] = request.max_tokens
        else:
            # Anthropic requires max_tokens
            data["max_tokens"] = 4096

        if request.temperature is not None:
            data["temperature"] = request.temperature
        if request.top_p is not None:
            data["top_p"] = request.top_p
        if request.stream:
            data["stream"] = request.stream

        # Add any extra parameters
        data.update(request.extra_params)

        return data

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion using Anthropic."""
        if not self.validate_model(request.model):
            raise ModelNotSupportedError(
                f"Model {request.model} is not supported by Anthropic provider"
            )

        try:
            # Prepare the request data
            data = self.prepare_request(request)

            # Make the API call
            response = await self.client.messages.create(**data)

            # Extract the response
            content = ""
            if response.content:
                content = " ".join(
                    [block.text for block in response.content if hasattr(block, "text")]
                )

            usage = (
                {
                    "prompt_tokens": response.usage.input_tokens
                    if response.usage
                    else 0,
                    "completion_tokens": response.usage.output_tokens
                    if response.usage
                    else 0,
                    "total_tokens": (
                        response.usage.input_tokens + response.usage.output_tokens
                    )
                    if response.usage
                    else 0,
                }
                if response.usage
                else None
            )

            return CompletionResponse(
                content=content,
                model=response.model,
                usage=usage,
                finish_reason=response.stop_reason,
                provider=self.provider_type,
            )

        except anthropic.AuthenticationError as e:
            raise AuthenticationError(f"Anthropic authentication failed: {e}")
        except anthropic.RateLimitError as e:
            raise ProviderError(f"Anthropic rate limit exceeded: {e}")
        except anthropic.APIError as e:
            raise ProviderError(f"Anthropic API error: {e}")
        except Exception as e:
            raise ProviderError(f"Anthropic provider error: {e}")

    async def stream_complete(self, request: CompletionRequest) -> AsyncIterator[str]:
        """Generate a streaming completion using Anthropic."""
        if not self.validate_model(request.model):
            raise ModelNotSupportedError(
                f"Model {request.model} is not supported by Anthropic provider"
            )

        try:
            # Prepare the request data
            data = self.prepare_request(request)
            # Remove stream parameter as it's not needed for the stream() method
            data.pop("stream", None)

            # Make the streaming API call
            async with self.client.messages.stream(**data) as stream:
                async for text in stream.text_stream:
                    yield text

        except anthropic.AuthenticationError as e:
            raise AuthenticationError(f"Anthropic authentication failed: {e}")
        except anthropic.RateLimitError as e:
            raise ProviderError(f"Anthropic rate limit exceeded: {e}")
        except anthropic.APIError as e:
            raise ProviderError(f"Anthropic API error: {e}")
        except Exception as e:
            raise ProviderError(f"Anthropic provider error: {e}")
