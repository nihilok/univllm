import pytest


@pytest.mark.parametrize(
    "model",
    [
        # "gpt-5-nano",
        # "claude-3-7-sonnet-latest",
        # "deepseek-chat",
        # "mistral-small-latest",
    ],
)
def test_all_providers(model):
    """
    Warning: This test will make real API calls. Ensure you have set the necessary
    environment variables for authentication (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
    and be aware of potential costs. Uncomment models in the parametrize list to test specific providers.
    """
    from univllm import UniversalLLMClient

    client = UniversalLLMClient()

    async def run_test():
        response = await client.complete(["Hello, how are you?"], model=model)
        print(f"Response from {model}:", response)
        assert response is not None

    import asyncio

    asyncio.run(run_test())


def test_vision_generate_image_openai():
    """Integration test: generate an image using OpenAI image model.

    This test is skipped automatically if OPENAI_API_KEY is not set.
    It requests a small 256x256 image to reduce cost and retrieves the
    image data in base64 format without writing to disk.
    """
    import os
    from univllm import UniversalLLMClient
    from univllm.exceptions import ProviderError, ModelNotSupportedError

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping real image generation test")

    client = UniversalLLMClient()
    model = "gpt-image-1"
    prompt = "A small, simple flat icon of a friendly robot head, minimal lines, white background"

    async def run_test():
        try:
            response = await client.generate_image(
                prompt=prompt,
                model=model,
                size="256x256",
                response_format="b64_json",
                n=1,  # extra param supported by OpenAI image API
            )
        except ModelNotSupportedError:
            pytest.skip(f"Model {model} not supported in current environment")
        except ProviderError as e:
            pytest.skip(f"Provider error during image generation: {e}")

        # Basic structure assertions
        assert response.model.startswith("gpt-image")
        assert response.prompt == prompt
        assert len(response.images) == 1
        img = response.images[0]
        # One of b64_json or url should be populated based on response_format
        assert img.b64_json is not None or img.url is not None
        if img.b64_json:
            # Spot check base64 length (should be reasonably large for even tiny images)
            assert len(img.b64_json) > 100

    import asyncio

    asyncio.run(run_test())


def test_image_generation_unsupported_model():
    """Test error handling for unsupported models/providers during image generation.

    Covers:
    1. Completely unsupported model name -> ModelNotSupportedError
    2. Supported provider & model (OpenAI gpt-4o) but not an image model -> ModelNotSupportedError
    3. Provider that does not implement image generation (Deepseek) -> NotImplementedError
    """
    import os
    import asyncio
    from univllm import UniversalLLMClient, ProviderType
    from univllm.exceptions import ModelNotSupportedError

    client = UniversalLLMClient()

    # 1. Completely unsupported model name (no provider autodetect)
    with pytest.raises(ModelNotSupportedError):
        asyncio.run(
            client.generate_image(
                prompt="test", model="totally-unknown-model"  # no gpt/claude/etc substring
            )
        )

    # 2. Non-image OpenAI model (requires API key or skip)
    if os.getenv("OPENAI_API_KEY"):
        with pytest.raises(ModelNotSupportedError):
            asyncio.run(
                client.generate_image(
                    prompt="icon", model="gpt-4o"  # valid OpenAI model but not image
                )
            )
    else:
        pytest.skip("Skipping OpenAI non-image model test: OPENAI_API_KEY not set")

    # 3. Provider without image generation implementation (Deepseek)
    # Use a separate client initialized with dummy Deepseek key to avoid env dependency
    deepseek_client = UniversalLLMClient(provider=ProviderType.DEEPSEEK, api_key="dummy")
    # generate_image will call BaseLLMProvider.generate_image -> NotImplementedError
    with pytest.raises(NotImplementedError):
        asyncio.run(
            deepseek_client.generate_image(
                prompt="robot", model="deepseek-chat", provider=ProviderType.DEEPSEEK
            )
        )
