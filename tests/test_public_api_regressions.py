"""Regression tests for the public client and provider contract."""

from unittest.mock import MagicMock

import pytest

from univllm import (
    AuthenticationError,
    BaseLLMProvider,
    CompletionResponse,
    ConfigurationError,
    ModelCapabilities,
    UniversalLLMClient,
)
from univllm.models import (
    CompletionRequest,
    ImageGenerationRequest,
    Message,
    MessageRole,
    ProviderType,
)
from univllm.providers.deepseek_provider import DeepseekProvider
from univllm.providers.mistral_provider import MistralProvider


def test_request_models_allow_unknown_model_names():
    """Model strings should not be rejected before provider selection."""
    request = CompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Hello")],
        model="claude-future-9",
    )
    image_request = ImageGenerationRequest(
        prompt="A lighthouse at dusk",
        model="image-future-1",
    )

    assert request.model == "claude-future-9"
    assert image_request.model == "image-future-1"


def test_public_api_exports_include_contract_types():
    """Public contract types should be importable from the package root."""
    assert AuthenticationError is not None
    assert BaseLLMProvider is not None
    assert CompletionResponse is not None
    assert ConfigurationError is not None
    assert ModelCapabilities is not None


def test_get_supported_models_is_available_as_classmethod():
    """Supported models should be queryable without instantiating a client."""
    models = UniversalLLMClient.get_supported_models()
    assert models


def test_get_model_capabilities_does_not_require_provider_initialization():
    """Capability lookup should not need an API key."""
    client = UniversalLLMClient()
    capabilities = client.get_model_capabilities("gpt-4o")

    assert capabilities.supports_streaming is True
    assert capabilities.context_window == 128000


@pytest.mark.asyncio
async def test_complete_rejects_raw_string_messages():
    """Raw string message lists should fail with a clear error."""
    client = UniversalLLMClient()

    with pytest.raises(ValueError, match="raw strings are not supported"):
        await client.complete(messages=["hello"], model="gpt-4o")


@pytest.mark.asyncio
async def test_stream_complete_rejects_raw_string_messages():
    """The streaming API should reject the same ambiguous raw string format."""
    client = UniversalLLMClient()

    with pytest.raises(ValueError, match="raw strings are not supported"):
        async for _ in client.stream_complete(messages=["hello"], model="gpt-4o"):
            pass


@pytest.mark.asyncio
async def test_provider_switch_requires_explicit_reconfiguration():
    """A bound client should not silently jump providers mid-call."""
    client = UniversalLLMClient(provider=ProviderType.OPENAI, api_key="openai-key")

    with pytest.raises(ConfigurationError, match="already initialized for openai"):
        await client.complete(
            messages=[{"role": "user", "content": "Hi"}],
            model="claude-sonnet-4-5",
        )


@pytest.mark.asyncio
async def test_stream_complete_forwards_tools_and_tool_choice(monkeypatch):
    """Streaming should accept the same tool parameters as non-streaming calls."""
    client = UniversalLLMClient(provider=ProviderType.OPENAI, api_key="test-key")
    provider = client.provider_instance
    assert provider is not None

    captured_kwargs = {}

    class _Stream:
        def __init__(self, chunks):
            self._chunks = iter(chunks)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._chunks)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = "hello"

    async def mock_create(**kwargs):
        captured_kwargs.update(kwargs)
        return _Stream([chunk])

    monkeypatch.setattr(provider.client.chat.completions, "create", mock_create)

    tool = {
        "name": "lookup_weather",
        "description": "Look up the weather",
        "input_schema": {"type": "object", "properties": {}},
    }

    chunks = [
        part
        async for part in client.stream_complete(
            messages=[{"role": "user", "content": "Weather?"}],
            model="gpt-4o",
            tools=[tool],
            tool_choice="auto",
        )
    ]

    assert chunks == ["hello"]
    assert captured_kwargs["tool_choice"] == "auto"
    assert captured_kwargs["tools"][0]["function"]["name"] == "lookup_weather"


@pytest.mark.asyncio
async def test_anthropic_text_blocks_are_not_space_joined(monkeypatch):
    """Adjacent text blocks should preserve exact provider output."""
    client = UniversalLLMClient(provider=ProviderType.ANTHROPIC, api_key="test-key")
    provider = client.provider_instance
    assert provider is not None

    first = MagicMock()
    first.type = "text"
    first.text = "Hello"

    second = MagicMock()
    second.type = "text"
    second.text = "world"

    response = MagicMock()
    response.content = [first, second]
    response.model = "claude-sonnet-4-5"
    response.stop_reason = "end_turn"
    response.usage = None

    async def mock_create(**kwargs):
        return response

    monkeypatch.setattr(provider.client.messages, "create", mock_create)

    result = await client.complete(
        messages=[{"role": "user", "content": "Hi"}],
        model="claude-sonnet-4-5",
    )

    assert result.content == "Helloworld"


def test_deepseek_uses_openai_sdk_client():
    """DeepSeek should use an OpenAI-compatible SDK client, not raw httpx."""
    provider = DeepseekProvider(api_key="test-key")
    assert hasattr(provider.client, "chat")
    assert hasattr(provider.client.chat, "completions")


def test_mistral_uses_mistral_sdk_client():
    """Mistral should use the official SDK client, not raw httpx."""
    provider = MistralProvider(api_key="test-key")
    assert hasattr(provider.client, "chat")
    assert hasattr(provider.client.chat, "complete_async")


@pytest.mark.asyncio
async def test_mistral_stream_complete_omits_stream_kwarg(monkeypatch):
    """The SDK stream method should not receive a redundant stream kwarg."""
    provider = MistralProvider(api_key="test-key")
    captured_kwargs = {}

    class _Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    async def mock_stream_async(**kwargs):
        captured_kwargs.update(kwargs)
        return _Stream()

    monkeypatch.setattr(provider.client.chat, "stream_async", mock_stream_async)

    request = CompletionRequest(
        messages=[Message(role=MessageRole.USER, content="Hi")],
        model="mistral-large-3",
        stream=True,
    )

    chunks = [chunk async for chunk in provider.stream_complete(request)]

    assert chunks == []
    assert "stream" not in captured_kwargs


@pytest.mark.asyncio
async def test_mistral_provider_uses_sdk_async_context_manager(monkeypatch):
    """Provider teardown should delegate to the SDK context manager surface."""
    provider = MistralProvider(api_key="test-key")
    calls = []

    async def mock_aenter():
        calls.append("enter")
        return provider.client

    async def mock_aexit(exc_type, exc_val, exc_tb):
        calls.append((exc_type, exc_val, exc_tb))

    monkeypatch.setattr(provider.client, "__aenter__", mock_aenter)
    monkeypatch.setattr(provider.client, "__aexit__", mock_aexit)

    await provider.__aenter__()
    await provider.__aexit__(None, None, None)

    assert calls == ["enter", (None, None, None)]
