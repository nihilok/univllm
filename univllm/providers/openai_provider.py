"""OpenAI provider implementation."""

import os
from typing import List, Optional, AsyncIterator
import openai

from ..supported_models import OPENAI_SUPPORTED_MODELS
from ..models import (
    CompletionRequest,
    CompletionResponse,
    ModelCapabilities,
    ProviderType,
)
from ..exceptions import ProviderError, ModelNotSupportedError, AuthenticationError
from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider for GPT models."""

    # Class-level supported models (prefixes or exact names)
    SUPPORTED_MODELS: List[str] = OPENAI_SUPPORTED_MODELS

    def __init__(self, api_key: Optional[str] = None, **kwargs) -> None:
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (if not provided, will use OPENAI_API_KEY env var)
            **kwargs: Additional configuration
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise AuthenticationError("OpenAI API key is required")

        super().__init__(api_key=api_key, **kwargs)
        self.client = openai.AsyncOpenAI(api_key=api_key)

    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.OPENAI

    def get_model_capabilities(self, model: str) -> ModelCapabilities:
        """Get capabilities for a specific OpenAI model."""
        if not self.validate_model(model):
            raise ModelNotSupportedError(
                f"Model {model} is not supported by OpenAI provider"
            )

        # Default capabilities for OpenAI models
        capabilities = ModelCapabilities(
            supports_system_messages=True,
            supports_function_calling=True,
            supports_streaming=True,
        )

        # Model-specific capabilities
        if "gpt-4" in model:
            capabilities.context_window = 8192 if "turbo" not in model else 128000
            capabilities.max_tokens = 4096
            if "vision" in model:
                capabilities.supports_vision = True
        elif "gpt-3.5-turbo" in model:
            capabilities.context_window = 4096
            capabilities.max_tokens = 4096

        return capabilities

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion using OpenAI."""
        if not self.validate_model(request.model):
            raise ModelNotSupportedError(
                f"Model {request.model} is not supported by OpenAI provider"
            )

        try:
            # Prepare the request data
            data = self.prepare_request(request)

            # Make the API call
            response = await self.client.chat.completions.create(**data)

            # Extract the response
            content = response.choices[0].message.content or ""
            usage = (
                {
                    "prompt_tokens": response.usage.prompt_tokens
                    if response.usage
                    else 0,
                    "completion_tokens": response.usage.completion_tokens
                    if response.usage
                    else 0,
                    "total_tokens": response.usage.total_tokens
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
                finish_reason=response.choices[0].finish_reason,
                provider=self.provider_type,
            )

        except openai.AuthenticationError as e:
            raise AuthenticationError(f"OpenAI authentication failed: {e}")
        except openai.RateLimitError as e:
            raise ProviderError(f"OpenAI rate limit exceeded: {e}")
        except openai.APIError as e:
            raise ProviderError(f"OpenAI API error: {e}")
        except Exception as e:
            raise ProviderError(f"OpenAI provider error: {e}")

    async def stream_complete(self, request: CompletionRequest) -> AsyncIterator[str]:
        """Generate a streaming completion using OpenAI."""
        if not self.validate_model(request.model):
            raise ModelNotSupportedError(
                f"Model {request.model} is not supported by OpenAI provider"
            )

        try:
            # Prepare the request data with streaming enabled
            data = self.prepare_request(request)
            data["stream"] = True

            # Make the streaming API call
            stream = await self.client.chat.completions.create(**data)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except openai.AuthenticationError as e:
            raise AuthenticationError(f"OpenAI authentication failed: {e}")
        except openai.RateLimitError as e:
            raise ProviderError(f"OpenAI rate limit exceeded: {e}")
        except openai.APIError as e:
            raise ProviderError(f"OpenAI API error: {e}")
        except Exception as e:
            raise ProviderError(f"OpenAI provider error: {e}")
