MISTRAL_SUPPORTED_MODELS = [
    "mistral-large-3",
    "mistral-medium-3",
    "mistral-small-3",
    "ministral-3-",
    "magistral-medium-",
    "magistral-small-",
    "codestral-",
    "devstral-",
    "voxtral-",
    "mistral-ocr-",
    "ocr-3-",
    # Legacy models
    "mistral-small-",
    "mistral-medium-",
]

OPENAI_SUPPORTED_MODELS = [
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-codex",
    "gpt-oss-120b",
    "gpt-oss-20b",
    "gpt-vision-1",
    "gpt-4o",
    "gpt-4",
    "gpt-image-1",  # image generation model
    "gpt-image",    # future variants
    "dall-e-2",     # added DALL-E 2
    "dall-e-3",     # added DALL-E 3
]

ANTHROPIC_SUPPORTED_MODELS = [
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-opus-4-",
    "claude-sonnet-4-",
    "claude-haiku-4-",
    "claude-code",
    # Legacy models
    "claude-3-7-sonnet-",
    "claude-3-5-sonnet-",
]

DEEPSEEK_SUPPORTED_MODELS = [
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-coder",
    "deepseek-vl",
    "deepseek-v3",
]


def is_potentially_supported_model(model_name: str) -> bool:
    all_supported_models = (
        MISTRAL_SUPPORTED_MODELS
        + OPENAI_SUPPORTED_MODELS
        + ANTHROPIC_SUPPORTED_MODELS
        + DEEPSEEK_SUPPORTED_MODELS
    )
    return any(model_name.startswith(prefix) for prefix in all_supported_models)


def is_unsupported_model(model_name: str) -> bool:
    return not is_potentially_supported_model(model_name)
