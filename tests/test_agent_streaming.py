"""Tests for streaming (added in 0.8.0).

Public contract:

* ``LLMProvider.stream(request)`` yields ``StreamChunk`` objects: zero+
  ``text`` chunks then exactly one ``final`` chunk carrying the
  assembled ``LLMResponse``.
* The base ``LLMProvider.stream`` is a non-streaming FALLBACK: it calls
  ``complete()`` and yields the whole content as one text chunk + the
  final. So every provider supports streaming, degraded or native.
* ``Agent.run_stream(prompt)`` / ``run_messages_stream(messages)`` yield
  ``StreamEvent`` objects: ``text`` deltas, ``tool_start`` / ``tool_end``
  around tool calls, ``correction`` from the post_turn_hook, and a final
  ``done`` (output + messages + steps) or ``error`` event.
* The full tool-use loop works in streaming mode exactly like
  ``run_messages``.
"""

from __future__ import annotations

from collections.abc import Iterator

from autoagent.agent import Agent, AgentTurnContext
from autoagent.providers.base import LLMProvider
from autoagent.schema import (
    LLMRequest,
    LLMResponse,
    Message,
    ModelConfig,
    StreamChunk,
    ToolCall,
    ToolSpec,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeNativeStreamProvider(LLMProvider):
    """Provider with NATIVE streaming: each scripted response is a list of
    text fragments + a final LLMResponse. Lets us assert that text deltas
    flow through and tool calls drive the loop."""

    def __init__(self, scripted: list[tuple[list[str], LLMResponse]]) -> None:
        super().__init__(ModelConfig(provider="fake", model="fake-stream"))
        self.scripted = scripted
        self.stream_calls = 0
        self.complete_calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover
        self.complete_calls += 1
        _fragments, response = self.scripted[0]
        return response

    def stream(self, request: LLMRequest) -> Iterator[StreamChunk]:
        self.stream_calls += 1
        fragments, response = self.scripted.pop(0)
        for frag in fragments:
            yield StreamChunk(type="text", text=frag)
        yield StreamChunk(type="final", response=response)


class FakeCompleteOnlyProvider(LLMProvider):
    """Provider WITHOUT a stream() override — exercises the base fallback."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(ModelConfig(provider="fake", model="fake-complete"))
        self.responses = responses

    def complete(self, request: LLMRequest) -> LLMResponse:
        return self.responses.pop(0)


def _spec(name: str) -> ToolSpec:
    return ToolSpec(name=name, description="d", input_schema={"type": "object", "properties": {}})


# ---------------------------------------------------------------------------
# Provider-level streaming
# ---------------------------------------------------------------------------


class TestProviderStream:
    def test_native_stream_yields_text_then_final(self) -> None:
        provider = FakeNativeStreamProvider(
            [(["Bon", "jour"], LLMResponse(content="Bonjour", model="x"))]
        )
        chunks = list(provider.stream(LLMRequest(messages=[])))
        assert [c.type for c in chunks] == ["text", "text", "final"]
        assert chunks[0].text == "Bon"
        assert chunks[1].text == "Jour".lower() or chunks[1].text == "jour"
        assert chunks[-1].response.content == "Bonjour"

    def test_base_fallback_yields_whole_then_final(self) -> None:
        provider = FakeCompleteOnlyProvider([LLMResponse(content="Hello world", model="x")])
        chunks = list(provider.stream(LLMRequest(messages=[])))
        assert [c.type for c in chunks] == ["text", "final"]
        assert chunks[0].text == "Hello world"
        assert chunks[1].response.content == "Hello world"

    def test_base_fallback_empty_content_skips_text_chunk(self) -> None:
        # An assistant turn that's pure tool calls has no text — the
        # fallback should not emit an empty text chunk.
        resp = LLMResponse(content="", tool_calls=[ToolCall(id="c", name="t")], model="x")
        provider = FakeCompleteOnlyProvider([resp])
        chunks = list(provider.stream(LLMRequest(messages=[])))
        assert [c.type for c in chunks] == ["final"]


# ---------------------------------------------------------------------------
# Agent-level streaming
# ---------------------------------------------------------------------------


class TestAgentRunStream:
    def test_simple_text_run(self) -> None:
        provider = FakeNativeStreamProvider(
            [(["Salut ", "Madame ", "Bernard"], LLMResponse(content="Salut Madame Bernard"))]
        )
        agent = Agent(provider=provider, system_prompt="x")
        events = list(agent.run_stream("bonjour"))

        text_events = [e for e in events if e.type == "text"]
        assert "".join(e.text for e in text_events) == "Salut Madame Bernard"

        done = events[-1]
        assert done.type == "done"
        assert done.output == "Salut Madame Bernard"
        assert done.steps == 1
        # The done event carries the full conversation for persistence.
        assert any(m.role == "assistant" for m in done.messages)

    def test_tool_loop(self) -> None:
        # Turn 1 : a tool call (no final text). Turn 2 : the answer.
        provider = FakeNativeStreamProvider(
            [
                ([], LLMResponse(content="", tool_calls=[ToolCall(id="c1", name="ping")])),
                (["pong ", "done"], LLMResponse(content="pong done")),
            ]
        )
        agent = Agent(provider=provider, system_prompt="x")

        @agent.tool(name="ping", description="d", input_schema={"type": "object", "properties": {}})
        def ping() -> str:
            return "pong"

        events = list(agent.run_stream("hi"))
        types = [e.type for e in events]
        assert "tool_start" in types
        assert "tool_end" in types
        # tool_start/end carry the tool name
        ts = next(e for e in events if e.type == "tool_start")
        assert ts.tool_name == "ping"
        te = next(e for e in events if e.type == "tool_end")
        assert te.tool_status == "ok"
        # final answer streamed after the tool round
        done = events[-1]
        assert done.type == "done"
        assert done.output == "pong done"
        assert done.steps == 2

    def test_post_turn_hook_correction(self) -> None:
        provider = FakeNativeStreamProvider(
            [
                (["c'est noté"], LLMResponse(content="c'est noté")),
                (["voici la vraie question"], LLMResponse(content="voici la vraie question")),
            ]
        )

        def hook(ctx: AgentTurnContext) -> Message | None:
            # Correct exactly once.
            if ctx.correction_count == 0:
                return Message(role="user", content="corrige-toi")
            return None

        agent = Agent(
            provider=provider,
            system_prompt="x",
            post_turn_hook=hook,
            max_corrections_per_run=1,
        )
        events = list(agent.run_stream("hi"))
        assert any(e.type == "correction" for e in events)
        corr = next(e for e in events if e.type == "correction")
        assert corr.text == "corrige-toi"
        assert events[-1].type == "done"
        assert events[-1].output == "voici la vraie question"

    def test_max_steps_yields_error(self) -> None:
        # Always returns a tool call → never terminates → max_steps hit.
        def infinite_script() -> list:
            return [
                ([], LLMResponse(content="", tool_calls=[ToolCall(id=f"c{i}", name="ping")]))
                for i in range(10)
            ]

        provider = FakeNativeStreamProvider(infinite_script())
        agent = Agent(provider=provider, system_prompt="x", max_steps=3)

        @agent.tool(name="ping", description="d", input_schema={"type": "object", "properties": {}})
        def ping() -> str:
            return "pong"

        events = list(agent.run_stream("hi"))
        assert events[-1].type == "error"
        assert "max_steps" in events[-1].error

    def test_fallback_provider_streams_through_agent(self) -> None:
        # A complete-only provider still works in run_stream via fallback.
        provider = FakeCompleteOnlyProvider([LLMResponse(content="réponse complète")])
        agent = Agent(provider=provider, system_prompt="x")
        events = list(agent.run_stream("hi"))
        text = "".join(e.text for e in events if e.type == "text")
        assert text == "réponse complète"
        assert events[-1].type == "done"

    def test_cancel_token_yields_error(self) -> None:
        import threading

        provider = FakeNativeStreamProvider(
            [(["x"], LLMResponse(content="x"))]
        )
        agent = Agent(provider=provider, system_prompt="x")
        token = threading.Event()
        token.set()  # already cancelled before the first step
        events = list(agent.run_stream("hi", cancel_token=token))
        assert events[-1].type == "error"
        assert events[-1].error == "cancelled"
