"""Gemini provider implementation."""

import os
from typing import List, Optional, AsyncIterator, Dict, Tuple
from google import genai
from google.genai import types

from ..supported_models import GEMINI_SUPPORTED_MODELS
from ..models import (
    CompletionRequest,
    CompletionResponse,
    ModelCapabilities,
    MessageRole,
    ProviderType,
    ToolCall,
)
from ..exceptions import ProviderError, ModelNotSupportedError, AuthenticationError
from .base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """Gemini provider for Google Gemini models."""

    SUPPORTED_MODELS: List[str] = GEMINI_SUPPORTED_MODELS

    def __init__(self, api_key: Optional[str] = None, **kwargs) -> None:
        """Initialize Gemini provider.

        Args:
            api_key: Gemini API key (if not provided, will use GEMINI_API_KEY env var)
            **kwargs: Additional configuration
        """
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise AuthenticationError("Gemini API key is required")

        super().__init__(api_key=api_key, **kwargs)
        self.client = genai.Client(api_key=api_key)

    @property
    def provider_type(self) -> ProviderType:
        """Return the provider type."""
        return ProviderType.GEMINI

    def _prepare_messages_and_config(
        self, request: CompletionRequest
    ) -> Tuple[List[Dict], types.GenerateContentConfig]:
        """Prepare messages and configuration for Gemini API.

        Args:
            request: Completion request

        Returns:
            Tuple of (messages_content, config)
        """
        # Separate system messages from other messages
        system_instruction = None
        messages_content = []

        for msg in request.messages:
            if msg.role == MessageRole.SYSTEM:
                # Use the last system message as system_instruction
                system_instruction = msg.content
            else:
                # Convert 'assistant' to 'model' for Gemini API
                role = "model" if msg.role == MessageRole.ASSISTANT else msg.role.value
                messages_content.append({"role": role, "parts": [{"text": msg.content}]})

        # Configure generation parameters
        config = types.GenerateContentConfig()
        if system_instruction:
            config.system_instruction = system_instruction
        if request.max_tokens is not None:
            config.max_output_tokens = request.max_tokens
        if request.temperature is not None:
            config.temperature = request.temperature
        if request.top_p is not None:
            config.top_p = request.top_p

        # Add tools if provided (convert MCP format to Gemini format)
        if request.tools:
            function_declarations = []
            for tool in request.tools:
                # Convert MCP tool definition to Gemini function declaration
                func_decl = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
                function_declarations.append(func_decl)
            
            # Create Tool object with function declarations
            gemini_tool = types.Tool(function_declarations=function_declarations)
            config.tools = [gemini_tool]
            
            # Handle tool_choice if specified
            if request.tool_choice:
                # Gemini uses different format for tool choice
                if request.tool_choice == "auto":
                    config.tool_config = types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode="AUTO"
                        )
                    )
                elif request.tool_choice == "none":
                    config.tool_config = types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode="NONE"
                        )
                    )
                else:
                    # Specific tool name - use ANY mode and filter with allowed_function_names
                    config.tool_config = types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode="ANY",
                            allowed_function_names=[request.tool_choice]
                        )
                    )

        return messages_content, config

    def get_model_capabilities(self, model: str) -> ModelCapabilities:
        """Get capabilities for a specific Gemini model."""
        if not self.validate_model(model):
            raise ModelNotSupportedError(
                f"Model {model} is not supported by Gemini provider"
            )

        # Default capabilities for Gemini models
        capabilities = ModelCapabilities(
            supports_system_messages=True,
            supports_function_calling=True,
            supports_streaming=True,
            supports_vision=False,
        )

        # Model-specific capabilities based on latest Gemini specifications
        if model.startswith("gemini-2.5-pro"):
            # Gemini 2.5 Pro - advanced reasoning
            capabilities.context_window = 1000000
            capabilities.max_tokens = 8192
            capabilities.supports_vision = True
        elif model.startswith("gemini-2.5-flash"):
            # Gemini 2.5 Flash - price-performance
            capabilities.context_window = 1000000
            capabilities.max_tokens = 8192
            capabilities.supports_vision = True
        elif model.startswith("gemini-2.0-flash"):
            # Gemini 2.0 Flash
            capabilities.context_window = 1000000
            capabilities.max_tokens = 8192
            capabilities.supports_vision = True
        elif model.startswith("gemini-1.5-pro"):
            # Gemini 1.5 Pro
            capabilities.context_window = 2000000
            capabilities.max_tokens = 8192
            capabilities.supports_vision = True
        elif model.startswith("gemini-1.5-flash"):
            # Gemini 1.5 Flash
            capabilities.context_window = 1000000
            capabilities.max_tokens = 8192
            capabilities.supports_vision = True

        return capabilities

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion using Gemini."""
        if not self.validate_model(request.model):
            raise ModelNotSupportedError(
                f"Model {request.model} is not supported by Gemini provider"
            )

        try:
            # Prepare messages and configuration
            messages_content, config = self._prepare_messages_and_config(request)

            # Make the API call using async interface
            response = await self.client.aio.models.generate_content(
                model=request.model,
                contents=messages_content,
                config=config,
            )

            # Extract the response content
            content = ""
            tool_calls = None
            
            # Check if response has candidates
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                
                # Extract content from parts
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    text_parts = []
                    function_call_parts = []
                    
                    for part in candidate.content.parts:
                        # Check for text content
                        if hasattr(part, "text") and part.text:
                            text_parts.append(str(part.text))  # Ensure it's a string
                        # Check for function call
                        elif hasattr(part, "function_call"):
                            function_call_parts.append(part.function_call)
                    
                    content = " ".join(text_parts) if text_parts else ""
                    
                    # Process function calls (tool calls)
                    if function_call_parts:
                        tool_calls = []
                        for fc in function_call_parts:
                            # Extract arguments from function call
                            args = {}
                            if hasattr(fc, "args") and fc.args:
                                # fc.args is typically a dict-like object
                                args = dict(fc.args) if hasattr(fc.args, "__iter__") else {}
                            
                            tool_calls.append(
                                ToolCall(
                                    id=getattr(fc, "id", None),
                                    name=fc.name,
                                    arguments=args
                                )
                            )
            else:
                # Fallback to simple text extraction
                content = response.text if hasattr(response, "text") else ""

            # Extract usage information if available
            usage = None
            if hasattr(response, "usage_metadata"):
                usage_metadata = response.usage_metadata
                usage = {
                    "prompt_tokens": getattr(usage_metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(
                        usage_metadata, "candidates_token_count", 0
                    ),
                    "total_tokens": getattr(usage_metadata, "total_token_count", 0),
                }

            finish_reason = None
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "finish_reason"):
                    finish_reason = str(candidate.finish_reason)

            return CompletionResponse(
                content=content,
                model=request.model,
                usage=usage,
                finish_reason=finish_reason,
                provider=self.provider_type,
                tool_calls=tool_calls,
            )

        except Exception as e:
            error_str = str(e).lower()
            if "api key" in error_str or "auth" in error_str or "401" in error_str:
                raise AuthenticationError(f"Gemini authentication failed: {e}")
            elif "quota" in error_str or "rate" in error_str or "429" in error_str:
                raise ProviderError(f"Gemini rate limit exceeded: {e}")
            else:
                raise ProviderError(f"Gemini provider error: {e}")

    async def stream_complete(self, request: CompletionRequest) -> AsyncIterator[str]:
        """Generate a streaming completion using Gemini."""
        if not self.validate_model(request.model):
            raise ModelNotSupportedError(
                f"Model {request.model} is not supported by Gemini provider"
            )

        try:
            # Prepare messages and configuration
            messages_content, config = self._prepare_messages_and_config(request)

            # Make the streaming API call using async interface
            response_stream = await self.client.aio.models.generate_content_stream(
                model=request.model,
                contents=messages_content,
                config=config,
            )

            async for chunk in response_stream:
                if hasattr(chunk, "text") and chunk.text:
                    yield chunk.text

        except Exception as e:
            error_str = str(e).lower()
            if "api key" in error_str or "auth" in error_str or "401" in error_str:
                raise AuthenticationError(f"Gemini authentication failed: {e}")
            elif "quota" in error_str or "rate" in error_str or "429" in error_str:
                raise ProviderError(f"Gemini rate limit exceeded: {e}")
            else:
                raise ProviderError(f"Gemini provider error: {e}")
