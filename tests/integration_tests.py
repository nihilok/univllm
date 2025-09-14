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
