"""Opt-in parallel tool execution (0.10.0).

When the model requests several tools in one turn and
``Agent(parallel_tool_calls=True)``, the calls run concurrently in a
thread pool. The transcript must stay DETERMINISTIC: tool messages are
appended in the model's call order, whatever the completion order.
"""

from __future__ import annotations

import time

from autoagent import Agent
from autoagent.schema import LLMRequest, LLMResponse, ModelConfig, ToolCall

_SLEEP = 0.15


class _ScriptedProvider:
    """Rejoue des réponses prédéfinies (pas de vrai LLM)."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.config = ModelConfig(provider="fake", model="fake-model")
        self._responses = list(responses)

    def complete(self, request: LLMRequest) -> LLMResponse:
        return self._responses.pop(0)

    def stream(self, request: LLMRequest):
        from autoagent.schema import StreamChunk

        response = self.complete(request)
        if response.content:
            yield StreamChunk(type="text", text=response.content)
        yield StreamChunk(type="final", response=response)


def _two_call_provider() -> _ScriptedProvider:
    return _ScriptedProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(id="c1", name="slow_a", arguments={}),
                    ToolCall(id="c2", name="slow_b", arguments={}),
                ],
            ),
            LLMResponse(content="done"),
        ]
    )


def _register_slow_tools(agent: Agent, log: list[str]) -> None:
    @agent.tool
    def slow_a() -> dict:
        """Sleep then answer."""
        time.sleep(_SLEEP)
        log.append("a")
        return {"who": "a"}

    @agent.tool
    def slow_b() -> dict:
        """Sleep then answer."""
        time.sleep(_SLEEP)
        log.append("b")
        return {"who": "b"}


def test_parallel_tools_cut_latency_and_keep_transcript_order() -> None:
    agent = Agent(_two_call_provider(), parallel_tool_calls=True)
    log: list[str] = []
    _register_slow_tools(agent, log)

    t0 = time.monotonic()
    result = agent.run("go")
    elapsed = time.monotonic() - t0

    # Two 0.15s tools concurrently: well under the 0.30s sequential floor.
    assert elapsed < 2 * _SLEEP * 0.95, f"pas de gain de latence ({elapsed:.3f}s)"
    tool_messages = [m for m in result.messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_messages] == ["c1", "c2"]  # ordre du modèle
    assert '"who": "a"' in tool_messages[0].content
    assert '"who": "b"' in tool_messages[1].content


def test_sequential_remains_the_default() -> None:
    agent = Agent(_two_call_provider())  # parallel_tool_calls absent -> False
    log: list[str] = []
    _register_slow_tools(agent, log)

    t0 = time.monotonic()
    result = agent.run("go")
    elapsed = time.monotonic() - t0

    assert elapsed >= 2 * _SLEEP * 0.95  # bien séquentiel
    assert log == ["a", "b"]
    assert [m.tool_call_id for m in result.messages if m.role == "tool"] == ["c1", "c2"]


def test_parallel_stream_events_are_ordered() -> None:
    agent = Agent(_two_call_provider(), parallel_tool_calls=True)
    _register_slow_tools(agent, [])

    events = list(agent.run_stream("go"))
    starts = [e.tool_name for e in events if e.type == "tool_start"]
    ends = [e.tool_name for e in events if e.type == "tool_end"]
    assert starts == ["slow_a", "slow_b"]
    assert ends == ["slow_a", "slow_b"]  # réordonnés sur l'ordre d'appel
    assert events[-1].type == "done"
