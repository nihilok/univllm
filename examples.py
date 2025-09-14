"""
Examples demonstrating how to use the univllm package.
"""

import asyncio
import os
from univllm import UniversalLLMClient, ProviderType


async def basic_completion_example():
    """Basic completion example with auto-detection."""
    client = UniversalLLMClient()

    # The client will auto-detect the provider based on the model name
    response = await client.complete(
        messages=["What is the capital of France?"],
        model="gpt-3.5-turbo",  # Will auto-detect OpenAI
    )

    print(f"Response: {response.content}")
    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")


async def explicit_provider_example():
    """Example with explicit provider selection."""
    # Initialize with specific provider
    client = UniversalLLMClient(
        provider=ProviderType.ANTHROPIC, api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    response = await client.complete(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing briefly."},
        ],
        model="claude-3-sonnet-20240229",
        max_tokens=150,
        temperature=0.7,
    )

    print(f"Response: {response.content}")


async def streaming_example():
    """Example of streaming completion."""
    client = UniversalLLMClient()

    print("Streaming response:")
    async for chunk in client.stream_complete(
        messages=["Tell me a short story about a robot."],
        model="gpt-3.5-turbo",
        max_tokens=200,
    ):
        print(chunk, end="", flush=True)
    print()  # New line after streaming


async def multi_provider_example():
    """Example using multiple providers."""
    client = UniversalLLMClient()

    # Get supported models for all providers
    all_models = client.get_supported_models()
    print("Supported models by provider:")
    for provider, models in all_models.items():
        print(f"{provider}: {models[:3]}...")  # Show first 3 models

    # Compare responses from different providers
    question = "What is machine learning?"

    # OpenAI
    openai_response = await client.complete(messages=[question], model="gpt-3.5-turbo")
    print(f"OpenAI: {openai_response.content[:100]}...")

    # Anthropic (if API key is available)
    try:
        anthropic_response = await client.complete(
            messages=[question], model="claude-3-haiku-20240307"
        )
        print(f"Anthropic: {anthropic_response.content[:100]}...")
    except Exception as e:
        print(f"Anthropic not available: {e}")


async def model_capabilities_example():
    """Example of checking model capabilities."""
    client = UniversalLLMClient()

    models_to_check = [
        "gpt-4",
        "claude-3-opus-20240229",
        "deepseek-chat",
        "mistral-large-latest",
    ]

    for model in models_to_check:
        try:
            capabilities = client.get_model_capabilities(model)
            print(f"\nModel: {model}")
            print(
                f"  Supports system messages: {capabilities.supports_system_messages}"
            )
            print(
                f"  Supports function calling: {capabilities.supports_function_calling}"
            )
            print(f"  Supports vision: {capabilities.supports_vision}")
            print(f"  Context window: {capabilities.context_window}")
            print(f"  Max tokens: {capabilities.max_tokens}")
        except Exception as e:
            print(f"Could not get capabilities for {model}: {e}")


async def error_handling_example():
    """Example of error handling."""
    client = UniversalLLMClient()

    try:
        # This will fail if no API key is set
        response = await client.complete(messages=["Hello"], model="gpt-3.5-turbo")
        print(f"Success: {response.content}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

    try:
        # This will fail with unsupported model
        response = await client.complete(messages=["Hello"], model="nonexistent-model")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")


async def main():
    """Run all examples."""
    print("=== Basic Completion Example ===")
    try:
        await basic_completion_example()
    except Exception as e:
        print(f"Skipped: {e}")

    print("\n=== Explicit Provider Example ===")
    try:
        await explicit_provider_example()
    except Exception as e:
        print(f"Skipped: {e}")

    print("\n=== Streaming Example ===")
    try:
        await streaming_example()
    except Exception as e:
        print(f"Skipped: {e}")

    print("\n=== Multi-Provider Example ===")
    try:
        await multi_provider_example()
    except Exception as e:
        print(f"Skipped: {e}")

    print("\n=== Model Capabilities Example ===")
    await model_capabilities_example()

    print("\n=== Error Handling Example ===")
    await error_handling_example()


if __name__ == "__main__":
    asyncio.run(main())
