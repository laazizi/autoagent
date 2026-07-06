from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from autoagent.schema import LLMRequest, LLMResponse, ModelConfig, StreamChunk


class LLMProvider(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config

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
