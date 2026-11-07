from __future__ import annotations

from autoagent.schema import ModelConfig

from .anthropic import AnthropicProvider
from .base import LLMProvider
from .gemini import GeminiProvider
from .gemini_live import GeminiLiveProvider
from .openai import DeepSeekProvider, GroqProvider, OpenAIProvider
from .routing import RoutingProvider


def create_provider(config: ModelConfig) -> LLMProvider:
    provider = config.provider.lower()
    if provider == "openai":
        return OpenAIProvider(config)
    if provider == "anthropic":
        return AnthropicProvider(config)
    if provider == "deepseek":
        return DeepSeekProvider(config)
    if provider == "groq":
        return GroqProvider(config)
    if provider in {"gemini", "google"}:
        return GeminiProvider(config)
    if provider in {"gemini_live", "gemini-live"}:
        return GeminiLiveProvider(config)
    raise ValueError(f"Unsupported provider: {config.provider}")


__all__ = [
    "AnthropicProvider",
    "DeepSeekProvider",
    "GeminiLiveProvider",
    "GeminiProvider",
    "GroqProvider",
    "LLMProvider",
    "OpenAIProvider",
    "RoutingProvider",
    "create_provider",
]
