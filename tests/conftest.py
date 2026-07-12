from __future__ import annotations

from typing import Any


class FakeLLMProvider:
    """Returns predefined responses in order. For testing Agent without real LLMs."""

    def __init__(self, responses: list[Any] | None = None) -> None:
        from autoagent.schema import LLMRequest, ModelConfig

        self.config = ModelConfig(provider="fake", model="fake-model")
        self.responses = responses or []
        self.calls: list[LLMRequest] = []

    def complete(self, request: Any) -> Any:
        from autoagent.schema import LLMResponse

        self.calls.append(request)
        if not self.responses:
            return LLMResponse(content="done", model="fake-model")

        next_response = self.responses.pop(0)
        if isinstance(next_response, str):
            return LLMResponse(content=next_response, model="fake-model")
        if isinstance(next_response, LLMResponse):
            return next_response
        raise TypeError(f"Unexpected response type: {type(next_response)}")
