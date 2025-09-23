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
    ImageGenerationRequest,
    ImageGenerationResponse,
    GeneratedImage,
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
            supports_vision=False,
        )

        # Model-specific capabilities based on latest OpenAI specifications
        if model.startswith("gpt-5"):
            # GPT-5 series - advanced capabilities
            capabilities.context_window = 200000
            capabilities.max_tokens = 16384
            capabilities.supports_vision = True
            if "mini" in model:
                capabilities.context_window = 128000
                capabilities.max_tokens = 8192
            elif "nano" in model:
                capabilities.context_window = 64000
                capabilities.max_tokens = 4096
        elif model.startswith("gpt-4o"):
            # GPT-4o series - optimized models
            capabilities.context_window = 128000
            capabilities.max_tokens = 16384
            capabilities.supports_vision = True
        elif model.startswith("gpt-oss-"):
            # Open source variants
            if "120b" in model:
                capabilities.context_window = 32000
                capabilities.max_tokens = 8192
            elif "20b" in model:
                capabilities.context_window = 16000
                capabilities.max_tokens = 4096
        elif model.startswith("gpt-vision-"):
            # Vision-specialized models
            capabilities.context_window = 128000
            capabilities.max_tokens = 8192
            capabilities.supports_vision = True
        elif model.startswith("gpt-image") or model.startswith("dall-e-"):
            # Image generation models - treat as vision capable
            capabilities.supports_vision = True

        return capabilities

    def prepare_request(self, request: CompletionRequest) -> dict:
        result = super().prepare_request(request)
        max_tokens = result.pop("max_tokens", None)
        if max_tokens:
            result["max_completion_tokens"] = max_tokens
        return result

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

    async def generate_image(
        self, request: ImageGenerationRequest
    ) -> ImageGenerationResponse:
        """Generate image(s) using OpenAI image models with size validation.

        Size rules (OpenAI as of Sep 2025):
          gpt-image-1: auto (default), 1024x1024, 1536x1024, 1024x1536
          dall-e-2: 256x256, 512x512, 1024x1024 (default 1024x1024 if omitted)
          dall-e-3: 1024x1024, 1792x1024, 1024x1792 (default 1024x1024 if omitted)
        """
        model = request.model
        if not self.validate_model(model):
            raise ModelNotSupportedError(
                f"Model {model} is not supported by OpenAI provider"
            )
        if not (model.startswith("gpt-image") or model.startswith("dall-e-")):
            raise ModelNotSupportedError(
                f"Model {model} is not an image generation model"
            )
        # Determine allowed sizes & defaults
        if model.startswith("gpt-image"):
            allowed = {"auto", "1024x1024", "1536x1024", "1024x1536"}
            default_size = "auto"
        elif model == "dall-e-2":
            allowed = {"256x256", "512x512", "1024x1024"}
            default_size = "1024x1024"
        elif model == "dall-e-3":
            allowed = {"1024x1024", "1792x1024", "1024x1792"}
            default_size = "1024x1024"
        else:  # future variants - don't enforce unless size provided
            allowed = set()
            default_size = None
        size = request.size or default_size
        if size and allowed and size not in allowed:
            raise ModelNotSupportedError(
                f"Invalid size '{size}' for model {model}. Allowed: {sorted(allowed)}"
            )
        try:
            payload = {
                "model": model,
                "prompt": request.prompt,
            }
            if size:
                payload["size"] = size
            # Only include response_format for DALL-E models; gpt-image-* currently rejects it.
            if model.startswith("dall-e-") and request.response_format:
                payload["response_format"] = request.response_format
            # Merge extra params last
            payload.update(request.extra_params)
            response = await self.client.images.generate(**payload)
            images: List[GeneratedImage] = []
            for item in getattr(response, "data", []):
                images.append(
                    GeneratedImage(
                        b64_json=getattr(item, "b64_json", None),
                        url=getattr(item, "url", None),
                    )
                )
            return ImageGenerationResponse(
                images=images,
                model=getattr(response, "model", model),
                provider=self.provider_type,
                created=getattr(response, "created", None),
                prompt=request.prompt,
            )
        except openai.AuthenticationError as e:
            raise AuthenticationError(f"OpenAI authentication failed: {e}")
        except openai.RateLimitError as e:
            raise ProviderError(f"OpenAI rate limit exceeded: {e}")
        except openai.APIError as e:
            raise ProviderError(f"OpenAI API error: {e}")
        except Exception as e:
            raise ProviderError(f"OpenAI image generation error: {e}")
