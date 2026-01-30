# univllm

[![PyPI version](https://badge.fury.io/py/univllm.svg)](https://badge.fury.io/py/univllm)

A universal Python package that provides a standardised interface for different LLM providers including OpenAI, Anthropic, Deepseek, and Mistral.

## Features

- **Universal Interface**: Single API to interact with multiple LLM providers
- **Auto-Detection**: Automatically detect the appropriate provider based on model name
- **Streaming Support**: Stream completions from all supported providers
- **Model Capabilities**: Query model capabilities like context window, function calling support, etc.
- **Error Handling**: Comprehensive error handling with provider-specific exceptions
- **Async Support**: Fully asynchronous API for better performance

## Supported Providers

- **OpenAI**: GPT-4o, GPT-5.x & GPT-5.2 family models
- **Anthropic**: Claude 3.x, 4.x & 4.5 family models  
- **Deepseek**: DeepSeek V3.2, Chat, Reasoner, Coder & VL models
- **Mistral**: Mistral Large 3, Ministral 3, Magistral, Codestral & specialized models

### Supported Model Prefixes
The library validates models using simple prefix matching (see `SUPPORTED_MODELS` lists). Any model string that begins with one of these prefixes will be accepted. Provider-specific suffixes or date/version tags (e.g. `-20240229`, `-latest`, `-0125`, minor patch tags) are allowed but not individually validated.

| Provider | Accepted Prefixes (Exact / Prefix Match)                                                                                                                                   | Notes                                                                                                     |
|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| OpenAI | `gpt-5.2`, `gpt-5.1`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-codex`, `gpt-oss-120b`, `gpt-oss-20b`, `gpt-vision-1`, `gpt-4o`, `gpt-4`                                  | GPT-5.2 is the latest flagship model (Dec 2025). Any extended suffix (e.g. `gpt-5.2-2025-12-11`) will pass if it starts with a listed prefix.             |
| Anthropic | `claude-opus-4-5`, `claude-sonnet-4-5`, `claude-haiku-4-5`, `claude-opus-4.1`, `claude-sonnet-4-`, `claude-haiku-4-`, `claude-opus-4-`, `claude-code`, `claude-3-7-sonnet-`, `claude-3-5-sonnet-` | Claude 4.5 series launched Sep-Nov 2025. Older variants (e.g. dated `claude-3-*` forms) can be added by extending the list in supported_models.py. |
| Deepseek | `deepseek-chat`, `deepseek-reasoner`, `deepseek-coder`, `deepseek-vl`, `deepseek-v3`                                                                                        | DeepSeek V3.2 models. `deepseek-reasoner` for advanced reasoning tasks. `deepseek-vl` for vision-language.                                                                                   |
| Mistral | `mistral-large-3`, `mistral-medium-3`, `mistral-small-3`, `ministral-3-`, `magistral-medium-`, `magistral-small-`, `codestral-`, `devstral-`, `voxtral-`, `mistral-ocr-`, `ocr-3-` | Mistral Large 3 (Dec 2025) flagship multimodal model. Ministral for edge, Codestral for code generation.                                                                                |

Note: If you need additional model prefixes, you can locally extend the corresponding `SUPPORTED_MODELS` list in `univllm/supported_models.py` or contribute a PR.

## Installation

```bash
pip install univllm
```

## Quick Start

```python
import asyncio
from univllm import UniversalLLMClient


async def main():
    client = UniversalLLMClient()

    # Auto-detects provider based on model name
    response = await client.complete(
        messages=["What is the capital of France?"],
        model="gpt-5.2"
    )

    print(response.content)


asyncio.run(main())
```

## Configuration

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
export MISTRAL_API_KEY="your-mistral-key"
```

Or pass them directly:

```python
from univllm import UniversalLLMClient, ProviderType

client = UniversalLLMClient(
    provider=ProviderType.OPENAI,
    api_key="your-api-key"
)
```

## Usage Examples

### Basic Completion

```python
import asyncio
from univllm import UniversalLLMClient


async def main():
    client = UniversalLLMClient()

    response = await client.complete(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing briefly."}
        ],
        model="gpt-5.2",
        max_tokens=150,
        temperature=0.7
    )

    print(f"Response: {response.content}")
    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")
    print(f"Usage: {response.usage}")


asyncio.run(main())
```

### Streaming Completion

```python
import asyncio
from univllm import UniversalLLMClient


async def main():
    client = UniversalLLMClient()

    async for chunk in client.stream_complete(
            messages=["Tell me a short story about a robot."],
            model="claude-sonnet-4-5",
            max_tokens=200
    ):
        print(chunk, end="", flush=True)


asyncio.run(main())
```

### Model Capabilities

```python
import asyncio
from univllm import UniversalLLMClient


async def main():
    client = UniversalLLMClient()

    # Get capabilities for a specific model
    capabilities = client.get_model_capabilities("gpt-5.2")

    print(f"Supports function calling: {capabilities.supports_function_calling}")
    print(f"Context window: {capabilities.context_window}")
    print(f"Max tokens: {capabilities.max_tokens}")

    # Get all supported models
    all_models = client.get_supported_models()
    for provider, models in all_models.items():
        print(f"{provider}: {len(models)} models")


asyncio.run(main())
```

### Multiple Providers

```python
import asyncio
from univllm import UniversalLLMClient
from univllm.models import ProviderType


async def main():
    client = UniversalLLMClient()

    question = "What is machine learning?"

    # OpenAI
    openai_response = await client.complete(
        messages=[question],
        model="gpt-5.2"
    )

    # Anthropic  
    anthropic_response = await client.complete(
        messages=[question],
        model="claude-sonnet-4-5"
    )

    print(f"OpenAI: {openai_response.content[:100]}...")
    print(f"Anthropic: {anthropic_response.content[:100]}...")


asyncio.run(main())
```

## API Reference

### UniversalLLMClient

Main client class for interacting with LLM providers.

#### Methods

- `complete()`: Generate a completion
- `stream_complete()`: Generate a streaming completion  
- `get_model_capabilities()`: Get model capabilities
- `get_supported_models()`: Get supported models for all providers
- `set_provider()`: Set or change the provider

### Models

- `CompletionRequest`: Request object for completions
- `CompletionResponse`: Response object from completions
- `ModelCapabilities`: Information about model capabilities
- `Message`: Individual message in a conversation

### Providers

- `ProviderType`: Enum of supported providers
- `BaseLLMProvider`: Base class for provider implementations

### Exceptions

- `UniversalLLMError`: Base exception
- `ProviderError`: Provider-related errors
- `ModelNotSupportedError`: Unsupported model errors
- `AuthenticationError`: Authentication failures
- `ConfigurationError`: Configuration issues

## Development

```bash
git clone https://github.com/nihilok/univllm.git
cd univllm
pip install -e ".[dev]"
```

Run tests:
```bash
pytest
```

## Licence

MIT Licence
