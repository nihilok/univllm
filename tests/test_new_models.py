"""Tests for new 2026 model support."""

import pytest
from univllm import UniversalLLMClient
from univllm.models import ProviderType
from univllm.exceptions import ModelNotSupportedError


def test_anthropic_4_5_models():
    """Test that Claude 4.5 models are supported."""
    client = UniversalLLMClient()

    # Test Claude 4.5 series
    assert client._auto_detect_provider("claude-opus-4-5") == ProviderType.ANTHROPIC
    assert client._auto_detect_provider("claude-sonnet-4-5") == ProviderType.ANTHROPIC
    assert client._auto_detect_provider("claude-haiku-4-5") == ProviderType.ANTHROPIC
    assert client._auto_detect_provider("claude-opus-4-5-20251101") == ProviderType.ANTHROPIC
    assert client._auto_detect_provider("claude-sonnet-4-5-20250929") == ProviderType.ANTHROPIC
    assert client._auto_detect_provider("claude-haiku-4-5-20251001") == ProviderType.ANTHROPIC


def test_openai_5_2_models():
    """Test that GPT-5.2 models are supported."""
    client = UniversalLLMClient()

    # Test GPT-5.2 series
    assert client._auto_detect_provider("gpt-5.2") == ProviderType.OPENAI
    assert client._auto_detect_provider("gpt-5.2-2025-12-11") == ProviderType.OPENAI
    assert client._auto_detect_provider("gpt-5.2-pro") == ProviderType.OPENAI
    assert client._auto_detect_provider("gpt-5.2-instant") == ProviderType.OPENAI
    assert client._auto_detect_provider("gpt-5.2-thinking") == ProviderType.OPENAI
    assert client._auto_detect_provider("gpt-5.2-codex") == ProviderType.OPENAI
    assert client._auto_detect_provider("gpt-5.1") == ProviderType.OPENAI
    assert client._auto_detect_provider("gpt-5-codex") == ProviderType.OPENAI


def test_mistral_3_models():
    """Test that Mistral 3 series models are supported."""
    client = UniversalLLMClient()

    # Test Mistral 3 series
    assert client._auto_detect_provider("mistral-large-3") == ProviderType.MISTRAL
    assert client._auto_detect_provider("mistral-medium-3") == ProviderType.MISTRAL
    assert client._auto_detect_provider("mistral-small-3") == ProviderType.MISTRAL
    assert client._auto_detect_provider("ministral-3-14b-instruct") == ProviderType.MISTRAL
    assert client._auto_detect_provider("ministral-3-8b-instruct") == ProviderType.MISTRAL
    assert client._auto_detect_provider("ministral-3-3b-instruct") == ProviderType.MISTRAL
    assert client._auto_detect_provider("codestral-25.01") == ProviderType.MISTRAL
    assert client._auto_detect_provider("devstral-2") == ProviderType.MISTRAL
    assert client._auto_detect_provider("voxtral-mini") == ProviderType.MISTRAL
    assert client._auto_detect_provider("ocr-3-premier") == ProviderType.MISTRAL


def test_deepseek_v3_models():
    """Test that DeepSeek V3.2 models are supported."""
    client = UniversalLLMClient()

    # Test DeepSeek V3 series
    assert client._auto_detect_provider("deepseek-chat") == ProviderType.DEEPSEEK
    assert client._auto_detect_provider("deepseek-reasoner") == ProviderType.DEEPSEEK
    assert client._auto_detect_provider("deepseek-coder") == ProviderType.DEEPSEEK
    assert client._auto_detect_provider("deepseek-vl") == ProviderType.DEEPSEEK
    assert client._auto_detect_provider("deepseek-v3") == ProviderType.DEEPSEEK
    assert client._auto_detect_provider("deepseek-v3.2") == ProviderType.DEEPSEEK
    assert client._auto_detect_provider("deepseek-v3-0324") == ProviderType.DEEPSEEK


def test_legacy_models_still_work():
    """Test that legacy models are still supported."""
    client = UniversalLLMClient()

    # Legacy Anthropic
    assert client._auto_detect_provider("claude-3-5-sonnet-20240229") == ProviderType.ANTHROPIC
    assert client._auto_detect_provider("claude-3-7-sonnet-20241022") == ProviderType.ANTHROPIC

    # Legacy OpenAI
    assert client._auto_detect_provider("gpt-4o") == ProviderType.OPENAI
    assert client._auto_detect_provider("gpt-4o-mini-2024-07-18") == ProviderType.OPENAI

    # Legacy Mistral
    assert client._auto_detect_provider("mistral-small-latest") == ProviderType.MISTRAL
    assert client._auto_detect_provider("mistral-medium-latest") == ProviderType.MISTRAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
