from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from autoagent.schema import LLMRequest, LLMResponse, ModelConfig, StreamChunk


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base``; mutates and returns ``base``.

    Nested dicts are merged key by key; anything else (scalars, lists) is
    replaced by the ``overlay`` value.
    """
    for key, value in overlay.items():
        current = base.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            deep_merge(current, value)
        else:
            base[key] = value
    return base


class LLMProvider(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config

    def _with_extra_body(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Deep-merge ``config.extra_body`` into a freshly built payload.

        Escape hatch for model-specific knobs ``LLMRequest`` does not map —
        Gemini's ``thinkingConfig``, OpenAI's ``reasoning_effort``… Every
        provider calls this on the way out of ``_build_payload``, so the
        mechanism is uniform and costs nothing when ``extra_body`` is empty.
        """
        if not self.config.extra_body:
            return payload
        return deep_merge(payload, self.config.extra_body)

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return the next model response for the agent loop."""

    def stream(self, request: LLMRequest) -> Iterator[StreamChunk]:
        """Yield incremental chunks for the next model response.

        Default implementation is a NON-STREAMING FALLBACK: it calls
        ``complete()`` and emits the whole content as a single ``text``
        chunk, then the ``final`` chunk. This keeps the streaming API
        uniform across every provider — those without native SSE
        support degrade gracefully (the user gets the answer in one
        shot instead of token-by-token, but everything still works).

        Providers that support native streaming (Anthropic, Gemini)
        override this to emit real token deltas.
        """
        response = self.complete(request)
        if response.content:
            yield StreamChunk(type="text", text=response.content)
        yield StreamChunk(type="final", response=response)
