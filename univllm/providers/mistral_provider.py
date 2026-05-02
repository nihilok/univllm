"""Mistral provider implementation."""

import json
import os
from typing import AsyncIterator, List, Optional

from mistralai.client import Mistral, errors as mistral_errors

from ..exceptions import AuthenticationError, ModelNotSupportedError, ProviderError
from ..models import (
    CompletionRequest,
    CompletionResponse,
    ModelCapabilities,
    ProviderType,
    ToolCall,
)
from ..supported_models import MISTRAL_SUPPORTED_MODELS
from .base import BaseLLMProvider


class MistralProvider(BaseLLMProvider):
    """Mistral provider for Mistral models."""

    SUPPORTED_MODELS: List[str] = MISTRAL_SUPPORTED_MODELS

    def __init__(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs
    ) -> None:
        """Initialize Mistral provider."""
        api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise AuthenticationError("Mistral API key is required")

        super().__init__(api_key=api_key, **kwargs)
        self.base_url = base_url or "https://api.mistral.ai"
        self.client = Mistral(api_key=api_key, server_url=self.base_url)

    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.MISTRAL

    @classmethod
    def get_model_capabilities(cls, model: str) -> ModelCapabilities:
        """Get capabilities for a specific Mistral model."""
        if not cls.supports_model(model):
            raise ModelNotSupportedError(
                f"Model {model} is not supported by Mistral provider"
            )

        capabilities = ModelCapabilities(
            supports_system_messages=True,
            supports_function_calling=True,
            supports_streaming=True,
            supports_vision=False,
        )

        if model.startswith("mistral-large-3"):
            capabilities.context_window = 256000
            capabilities.max_tokens = 8192
            capabilities.supports_vision = True
        elif model.startswith("mistral-medium-3"):
            capabilities.context_window = 128000
            capabilities.max_tokens = 8192
            capabilities.supports_vision = True
        elif model.startswith("mistral-small-3"):
            capabilities.context_window = 128000
            capabilities.max_tokens = 8192
            capabilities.supports_vision = True
        elif model.startswith("ministral-3-"):
            capabilities.context_window = 128000
            capabilities.max_tokens = 8192
            capabilities.supports_vision = True
        elif model.startswith("magistral-medium-"):
            capabilities.context_window = 128000
            capabilities.max_tokens = 8192
        elif model.startswith("magistral-small-"):
            capabilities.context_window = 64000
            capabilities.max_tokens = 8192
        elif model.startswith("codestral-"):
            capabilities.context_window = 64000
            capabilities.max_tokens = 8192
        elif model.startswith("devstral-"):
            capabilities.context_window = 64000
            capabilities.max_tokens = 8192
        elif model.startswith("voxtral-"):
            capabilities.context_window = 64000
            capabilities.max_tokens = 8192
        elif model.startswith("mistral-ocr-") or model.startswith("ocr-3-"):
            capabilities.context_window = 64000
            capabilities.max_tokens = 8192
            capabilities.supports_vision = True
        elif model.startswith("mistral-small-") or model.startswith("mistral-medium-"):
            capabilities.context_window = 32000
            capabilities.max_tokens = 8192

        return capabilities

    def prepare_request(self, request: CompletionRequest) -> dict:
        """Prepare request data for the Mistral SDK."""
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

    @staticmethod
    def _coerce_text_content(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(part.text for part in content if hasattr(part, "text"))
        return str(content) if content else ""

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion using Mistral."""
        if not self.validate_model(request.model):
            raise ModelNotSupportedError(
                f"Model {request.model} is not supported by Mistral provider"
            )

        try:
            response = await self.client.chat.complete_async(**self.prepare_request(request))
            choice = response.choices[0] if response.choices else None
            message = choice.message if choice else None
            content = self._coerce_text_content(getattr(message, "content", ""))

            tool_calls = None
            if message and getattr(message, "tool_calls", None):
                tool_calls = []
                for tc in message.tool_calls:
                    raw_arguments = getattr(tc.function, "arguments", None)
                    if isinstance(raw_arguments, str):
                        try:
                            arguments = json.loads(raw_arguments)
                        except json.JSONDecodeError as e:
                            arguments = {"_parse_error": str(e), "_raw": raw_arguments}
                    else:
                        arguments = raw_arguments or {}

                    tool_calls.append(
                        ToolCall(
                            id=getattr(tc, "id", None),
                            name=tc.function.name,
                            arguments=arguments,
                        )
                    )

            usage = getattr(response, "usage", None)
            return CompletionResponse(
                content=content,
                model=getattr(response, "model", request.model),
                usage={
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }
                if usage
                else None,
                finish_reason=getattr(choice, "finish_reason", None),
                provider=self.provider_type,
                tool_calls=tool_calls,
            )

        except mistral_errors.SDKError as e:
            status_code = getattr(getattr(e, "raw_response", None), "status_code", None)
            if status_code == 401:
                raise AuthenticationError(f"Mistral authentication failed: {e}")
            if status_code == 429:
                raise ProviderError(f"Mistral rate limit exceeded: {e}")
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
            data = self.prepare_request(request)
            data.pop("stream", None)
            stream = await self.client.chat.stream_async(**data)

            async for event in stream:
                chunk = getattr(event, "data", None)
                if not chunk or not getattr(chunk, "choices", None):
                    continue

                delta = chunk.choices[0].delta
                text = self._coerce_text_content(getattr(delta, "content", None))
                if text:
                    yield text

        except mistral_errors.SDKError as e:
            status_code = getattr(getattr(e, "raw_response", None), "status_code", None)
            if status_code == 401:
                raise AuthenticationError(f"Mistral authentication failed: {e}")
            if status_code == 429:
                raise ProviderError(f"Mistral rate limit exceeded: {e}")
            raise ProviderError(f"Mistral API error: {e}")
        except Exception as e:
            raise ProviderError(f"Mistral provider error: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.__aexit__(exc_type, exc_val, exc_tb)
