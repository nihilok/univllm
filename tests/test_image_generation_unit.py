"""Unit tests for OpenAI image generation payload construction.

These tests ensure:
- gpt-image-* models omit response_format
- dall-e-* models include response_format
- Default sizes are applied correctly when size not provided
"""

import asyncio
import pytest

from univllm import UniversalLLMClient, ProviderType
from univllm.exceptions import ModelNotSupportedError


class _Capture:
    def __init__(self):
        self.calls = []

    async def fake_generate(self, **kwargs):  # signature matches images.generate(**payload)
        self.calls.append(kwargs)
        # Minimal fake response object replicating attributes accessed in provider
        class _Item:
            def __init__(self):
                # Simulate base64 field optionally returned
                self.b64_json = "ZmFrZV9pbWFnZV9kYXRh"  # base64 for 'fake_image_data'
                self.url = None
        class _Resp:
            def __init__(self):
                self.data = [_Item()]
                self.model = kwargs.get("model", "unknown")
                self.created = 1234567890
        return _Resp()


@pytest.mark.asyncio
async def test_openai_image_generation_payload_gpt_image_omits_response_format(monkeypatch):
    client = UniversalLLMClient(provider=ProviderType.OPENAI, api_key="test_key")
    provider = client.provider_instance  # type: ignore

    cap = _Capture()
    # Patch only the images.generate method
    monkeypatch.setattr(provider.client.images, "generate", cap.fake_generate)  # type: ignore[attr-defined]

    # Call without size -> default 'auto' expected, no response_format
    resp = await client.generate_image(
        prompt="Simple icon", model="gpt-image-1"
    )
    assert resp.model.startswith("gpt-image")
    assert len(cap.calls) == 1
    payload = cap.calls[0]
    assert payload["model"] == "gpt-image-1"
    # Size should default to 'auto'
    assert payload.get("size") == "auto"
    # response_format must be absent
    assert "response_format" not in payload

    # Second call with explicit valid size should include size but still exclude response_format
    resp2 = await client.generate_image(
        prompt="Simple icon 2", model="gpt-image-1", size="1024x1024"
    )
    assert resp2.model.startswith("gpt-image")
    assert len(cap.calls) == 2
    payload2 = cap.calls[1]
    assert payload2.get("size") == "1024x1024"
    assert "response_format" not in payload2


@pytest.mark.asyncio
async def test_openai_image_generation_payload_dalle_includes_response_format(monkeypatch):
    client = UniversalLLMClient(provider=ProviderType.OPENAI, api_key="test_key")
    provider = client.provider_instance  # type: ignore

    cap = _Capture()
    monkeypatch.setattr(provider.client.images, "generate", cap.fake_generate)  # type: ignore[attr-defined]

    # Call dall-e-2 with default settings
    resp = await client.generate_image(
        prompt="A sketch of a tree", model="dall-e-2"
    )
    assert resp.model.startswith("dall-e")
    assert len(cap.calls) == 1
    payload = cap.calls[0]
    assert payload["model"] == "dall-e-2"
    # Default size should be applied
    assert payload.get("size") == "1024x1024"
    # response_format should be present (default b64_json)
    assert payload.get("response_format") == "b64_json"

    # Invalid size should raise error
    with pytest.raises(ModelNotSupportedError):
        await client.generate_image(
            prompt="bad size", model="dall-e-2", size="999x999"
        )

    # Call with explicit alternative allowed size
    resp2 = await client.generate_image(
        prompt="Another tree", model="dall-e-2", size="512x512"
    )
    assert len(cap.calls) == 2  # second successful call recorded
    payload2 = cap.calls[1]
    assert payload2.get("size") == "512x512"
    assert payload2.get("response_format") == "b64_json"

