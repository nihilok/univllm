"""Basic tests for univllm package."""

import pytest
from univllm import UniversalLLMClient
from univllm.models import ModelCapabilities, MessageRole, Message, ProviderType
from univllm.exceptions import ModelNotSupportedError


def test_imports():
    """Test that all main classes can be imported."""
    from univllm import (
        UniversalLLMClient,
        ProviderType,
        UniversalLLMError,
        ProviderError,
        ModelNotSupportedError,
    )

    assert UniversalLLMClient is not None
    assert ProviderType is not None


def test_client_creation():
    """Test that client can be created without errors."""
    client = UniversalLLMClient()
    assert client is not None


def test_auto_detection():
    """Test model provider auto-detection."""
    client = UniversalLLMClient()

    # Test OpenAI models
    assert client._auto_detect_provider("gpt-3.5-turbo") == ProviderType.OPENAI
    assert client._auto_detect_provider("gpt-4") == ProviderType.OPENAI

    # Test Anthropic models
    assert (
        client._auto_detect_provider("claude-3-sonnet-20240229")
        == ProviderType.ANTHROPIC
    )
    assert client._auto_detect_provider("claude-2.1") == ProviderType.ANTHROPIC

    # Test Deepseek models
    assert client._auto_detect_provider("deepseek-chat") == ProviderType.DEEPSEEK
    assert client._auto_detect_provider("deepseek-coder") == ProviderType.DEEPSEEK

    # Test Mistral models
    assert client._auto_detect_provider("mistral-large-latest") == ProviderType.MISTRAL
    assert client._auto_detect_provider("mixtral-8x7b") == ProviderType.MISTRAL


def test_unsupported_model():
    """Test handling of unsupported models."""
    client = UniversalLLMClient()

    with pytest.raises(ModelNotSupportedError):
        client._auto_detect_provider("nonexistent-model")


def test_provider_types():
    """Test provider type enum values."""
    assert ProviderType.OPENAI == "openai"
    assert ProviderType.ANTHROPIC == "anthropic"
    assert ProviderType.DEEPSEEK == "deepseek"
    assert ProviderType.MISTRAL == "mistral"


def test_message_creation():
    """Test message model creation."""
    msg = Message(role=MessageRole.USER, content="Hello")
    assert msg.role == MessageRole.USER
    assert msg.content == "Hello"


def test_model_capabilities():
    """Test model capabilities structure."""
    caps = ModelCapabilities(
        supports_system_messages=True,
        supports_function_calling=False,
        supports_vision=True,
        context_window=4096,
    )
    assert caps.supports_system_messages is True
    assert caps.supports_function_calling is False
    assert caps.supports_vision is True
    assert caps.context_window == 4096


def test_supported_models_structure():
    """Test that get_supported_models returns proper structure."""
    client = UniversalLLMClient()
    models = client.get_supported_models()

    # Should return a dict with all provider types
    assert isinstance(models, dict)
    assert ProviderType.OPENAI in models
    assert ProviderType.ANTHROPIC in models
    assert ProviderType.DEEPSEEK in models
    assert ProviderType.MISTRAL in models

    # Each value should be a list
    for provider, model_list in models.items():
        assert isinstance(model_list, list)


if __name__ == "__main__":
    pytest.main([__file__])
