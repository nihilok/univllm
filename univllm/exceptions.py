"""Custom exceptions for univllm package."""


class UniversalLLMError(Exception):
    """Base exception for univllm package."""

    pass


class ProviderError(UniversalLLMError):
    """Exception raised when there's an error with a provider."""

    pass


class ModelNotSupportedError(UniversalLLMError):
    """Exception raised when a model is not supported by a provider."""

    pass


class ConfigurationError(UniversalLLMError):
    """Exception raised when there's a configuration issue."""

    pass


class AuthenticationError(UniversalLLMError):
    """Exception raised when authentication fails."""

    pass
