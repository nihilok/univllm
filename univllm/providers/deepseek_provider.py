"""Deepseek provider implementation."""

import json
import os
from typing import AsyncIterator, List, Optional

import openai

from ..exceptions import AuthenticationError, ModelNotSupportedError, ProviderError
from ..models import (
    CompletionRequest,
    CompletionResponse,
    ModelCapabilities,
    ProviderType,
    ToolCall,
)
from ..supported_models import DEEPSEEK_SUPPORTED_MODELS
from .base import BaseLLMProvider


class DeepseekProvider(BaseLLMProvider):
    """Deepseek provider for Deepseek models."""

    SUPPORTED_MODELS: List[str] = DEEPSEEK_SUPPORTED_MODELS

    def __init__(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs
    ) -> None:
        """Initialize Deepseek provider."""
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise AuthenticationError("Deepseek API key is required")

        super().__init__(api_key=api_key, **kwargs)
        self.base_url = self._normalize_base_url(base_url)
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=self.base_url)

    @staticmethod
    def _normalize_base_url(base_url: Optional[str]) -> str:
        normalized = (base_url or "https://api.deepseek.com/v1").rstrip("/")
        if not normalized.endswith("/v1"):
            normalized = f"{normalized}/v1"
        return normalized

    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.DEEPSEEK

    @classmethod
    def get_model_capabilities(cls, model: str) -> ModelCapabilities:
        """Get capabilities for a specific Deepseek model."""
        if not cls.supports_model(model):
            raise ModelNotSupportedError(
                f"Model {model} is not supported by Deepseek provider"
            )

        capabilities = ModelCapabilities(
            supports_system_messages=True,
            supports_function_calling=True,
            supports_streaming=True,
            supports_vision=False,
            context_window=64000,
            max_tokens=8192,
        )

        if model.startswith("deepseek-v4-flash"):
            capabilities.context_window = 128000
            capabilities.max_tokens = 8192
        elif model.startswith("deepseek-v4-pro"):
            capabilities.context_window = 128000
            capabilities.max_tokens = 16384
        elif model.startswith("deepseek-chat") or model.startswith("deepseek-v3"):
            capabilities.context_window = 128000
            capabilities.max_tokens = 8192
        elif model.startswith("deepseek-reasoner"):
            capabilities.context_window = 128000
            capabilities.max_tokens = 16384
        elif model.startswith("deepseek-coder"):
            capabilities.context_window = 64000
            capabilities.max_tokens = 8192
        elif model.startswith("deepseek-vl"):
            capabilities.context_window = 64000
            capabilities.max_tokens = 8192
            capabilities.supports_vision = True

        return capabilities

    def prepare_request(self, request: CompletionRequest) -> dict:
        """Prepare request data for DeepSeek's OpenAI-compatible API."""
        result = super().prepare_request(request)

        if request.tools:
            result["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in request.tools
            ]
            if request.tool_choice:
                result["tool_choice"] = request.tool_choice

        return result

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion using Deepseek."""
        if not self.validate_model(request.model):
            raise ModelNotSupportedError(
                f"Model {request.model} is not supported by Deepseek provider"
            )

        try:
            response = await self.client.chat.completions.create(
                **self.prepare_request(request)
            )
            message = response.choices[0].message
            content = message.content or ""

            tool_calls = None
            if getattr(message, "tool_calls", None):
                tool_calls = []
                for tc in message.tool_calls:
                    args = {}
                    if tc.function.arguments:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError as e:
                            args = {"_parse_error": str(e), "_raw": tc.function.arguments}

                    tool_calls.append(
                        ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                    )

            usage = (
                {
                    "prompt_tokens": response.usage.prompt_tokens
                    if response.usage
                    else 0,
                    "completion_tokens": response.usage.completion_tokens
                    if response.usage
                    else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
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
                tool_calls=tool_calls,
            )

        except openai.AuthenticationError as e:
            raise AuthenticationError(f"Deepseek authentication failed: {e}")
        except openai.RateLimitError as e:
            raise ProviderError(f"Deepseek rate limit exceeded: {e}")
        except openai.APIError as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if status_code == 401:
                raise AuthenticationError(f"Deepseek authentication failed: {e}")
            if status_code == 429:
                raise ProviderError(f"Deepseek rate limit exceeded: {e}")
            raise ProviderError(f"Deepseek API error: {e}")
        except Exception as e:
            raise ProviderError(f"Deepseek provider error: {e}")

    async def stream_complete(self, request: CompletionRequest) -> AsyncIterator[str]:
        """Generate a streaming completion using Deepseek."""
        if not self.validate_model(request.model):
            raise ModelNotSupportedError(
                f"Model {request.model} is not supported by Deepseek provider"
            )

        try:
            data = self.prepare_request(request)
            data["stream"] = True
            stream = await self.client.chat.completions.create(**data)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except openai.AuthenticationError as e:
            raise AuthenticationError(f"Deepseek authentication failed: {e}")
        except openai.RateLimitError as e:
            raise ProviderError(f"Deepseek rate limit exceeded: {e}")
        except openai.APIError as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if status_code == 401:
                raise AuthenticationError(f"Deepseek authentication failed: {e}")
            if status_code == 429:
                raise ProviderError(f"Deepseek rate limit exceeded: {e}")
            raise ProviderError(f"Deepseek API error: {e}")
        except Exception as e:
            raise ProviderError(f"Deepseek provider error: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.close()
