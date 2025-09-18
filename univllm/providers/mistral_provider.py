"""Mistral provider implementation."""

import os
import json
from typing import List, Optional, AsyncIterator
import httpx

from ..supported_models import MISTRAL_SUPPORTED_MODELS
from ..models import (
    CompletionRequest,
    CompletionResponse,
    ModelCapabilities,
    ProviderType,
)
from ..exceptions import ProviderError, ModelNotSupportedError, AuthenticationError
from .base import BaseLLMProvider


class MistralProvider(BaseLLMProvider):
    """Mistral provider for Mistral models."""

    SUPPORTED_MODELS: List[str] = MISTRAL_SUPPORTED_MODELS

    def __init__(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs
    ) -> None:
        """Initialize Mistral provider.

        Args:
            api_key: Mistral API key (if not provided, will use MISTRAL_API_KEY env var)
            base_url: Base URL for Mistral API (default: https://api.mistral.ai)
            **kwargs: Additional configuration
        """
        api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise AuthenticationError("Mistral API key is required")

        super().__init__(api_key=api_key, **kwargs)
        self.base_url = base_url or "https://api.mistral.ai"
        self.client = httpx.AsyncClient()

    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.MISTRAL

    def get_model_capabilities(self, model: str) -> ModelCapabilities:
        """Get capabilities for a specific Mistral model."""
        if not self.validate_model(model):
            raise ModelNotSupportedError(
                f"Model {model} is not supported by Mistral provider"
            )

        # Default capabilities for Mistral models
        capabilities = ModelCapabilities(
            supports_system_messages=True,
            supports_function_calling=True,
            supports_streaming=True,
            supports_vision=False,
        )

        # Model-specific capabilities based on latest Mistral specifications
        if model.startswith("mistral-small-"):
            # Mistral Small series - efficient models
            capabilities.context_window = 32000
            capabilities.max_tokens = 8192
        elif model.startswith("mistral-medium-"):
            # Mistral Medium series - balanced performance
            capabilities.context_window = 32000
            capabilities.max_tokens = 8192
        elif model.startswith("magistral-small-"):
            # Magistral Small series - specialized models
            capabilities.context_window = 32000
            capabilities.max_tokens = 8192
        elif model.startswith("magistral-medium-"):
            # Magistral Medium series - enhanced capabilities
            capabilities.context_window = 32000
            capabilities.max_tokens = 8192
        elif model.startswith("codestral-"):
            # Codestral series - code-specialized models
            capabilities.context_window = 32000
            capabilities.max_tokens = 8192
            capabilities.supports_function_calling = True  # Enhanced for code generation
        elif model.startswith("mistral-ocr-"):
            # Mistral OCR series - vision-capable models
            capabilities.context_window = 32000
            capabilities.max_tokens = 8192
            capabilities.supports_vision = True

        return capabilities

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion using Mistral."""
        if not self.validate_model(request.model):
            raise ModelNotSupportedError(
                f"Model {request.model} is not supported by Mistral provider"
            )

        try:
            # Prepare the request data
            data = self.prepare_request(request)

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Make the API call
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions", json=data, headers=headers
            )
            response.raise_for_status()

            result = response.json()

            # Extract the response
            content = (
                result["choices"][0]["message"]["content"]
                if result.get("choices")
                else ""
            )
            usage = result.get("usage", {})

            return CompletionResponse(
                content=content,
                model=result.get("model", request.model),
                usage={
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
                if usage
                else None,
                finish_reason=result["choices"][0].get("finish_reason")
                if result.get("choices")
                else None,
                provider=self.provider_type,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise AuthenticationError(f"Mistral authentication failed: {e}")
            elif e.response.status_code == 429:
                raise ProviderError(f"Mistral rate limit exceeded: {e}")
            else:
                raise ProviderError(f"Mistral API error: {e}")
        except Exception as e:
            raise ProviderError(f"Mistral provider error: {e}")

    async def stream_complete(self, request: CompletionRequest) -> AsyncIterator[str]:
        """Generate a streaming completion using Mistral."""
        if not self.validate_model(request.model):
            raise ModelNotSupportedError(
                f"Model {request.model} is not supported by Mistral provider"
            )

        try:
            # Prepare the request data with streaming enabled
            data = self.prepare_request(request)
            data["stream"] = True

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            # Make the streaming API call
            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=data,
                headers=headers,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk_data = line[6:]  # Remove "data: " prefix
                        if chunk_data.strip() == "[DONE]":
                            break

                        try:
                            chunk = json.loads(chunk_data)
                            if chunk.get("choices") and chunk["choices"][0].get(
                                "delta", {}
                            ).get("content"):
                                yield chunk["choices"][0]["delta"]["content"]
                        except json.JSONDecodeError:
                            continue

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise AuthenticationError(f"Mistral authentication failed: {e}")
            elif e.response.status_code == 429:
                raise ProviderError(f"Mistral rate limit exceeded: {e}")
            else:
                raise ProviderError(f"Mistral API error: {e}")
        except Exception as e:
            raise ProviderError(f"Mistral provider error: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()
