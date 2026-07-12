"""Tests for `autoagent.memory` and its wiring into `Agent` (0.6.0).

Public contract:

* `Memory` is a Protocol with two methods: `compact(messages)` and
  `recall(query, k)`. Anything implementing both can be passed to
  `Agent(memory=...)`.
* `BufferMemory(max_messages=N)` keeps system messages + last N
  non-system messages, with safe truncation (no orphan `tool` head).
* When `Agent(memory=None)` (the default), behaviour is unchanged
  vs prior versions.
* When `memory` is configured, `agent.run_messages` compacts the
  input ONCE before the loop. A buggy `compact` is isolated — the
  run proceeds with the original messages.
* `agent.register_recall_tool()` registers a `recall` tool only when
  a memory is configured; it is a silent no-op otherwise.
"""

from __future__ import annotations

from typing import Any

import pytest

from autoagent.agent import Agent
from autoagent.memory import BufferMemory, Memory
from autoagent.schema import LLMResponse, Message, ToolCall, ToolSpec

from .conftest import FakeLLMProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="fake")


def _msg(role: str, content: str = "x") -> Message:
    return Message(role=role, content=content)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Protocol shape — anything with .compact + .recall is a Memory
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_buffer_memory_satisfies_protocol(self) -> None:
        mem = BufferMemory(max_messages=5)
        assert isinstance(mem, Memory)

    def test_arbitrary_class_with_methods_satisfies_protocol(self) -> None:
        class Custom:
            def compact(self, messages: list[Message]) -> list[Message]:
                return messages

            def recall(self, query: str, k: int = 5) -> list[Message]:
                return []

        assert isinstance(Custom(), Memory)

    def test_class_missing_method_does_not_satisfy(self) -> None:
        class Partial:
            def compact(self, messages: list[Message]) -> list[Message]:
                return messages

        assert not isinstance(Partial(), Memory)


# ---------------------------------------------------------------------------
# BufferMemory behaviour
# ---------------------------------------------------------------------------


class TestBufferMemoryConstructor:
    def test_default_is_20(self) -> None:
        assert BufferMemory().max_messages == 20

    def test_rejects_zero(self) -> None:
        with pytest.raises(ValueError):
            BufferMemory(max_messages=0)

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            BufferMemory(max_messages=-3)


class TestBufferMemoryCompact:
    def test_short_history_unchanged(self) -> None:
        mem = BufferMemory(max_messages=10)
        msgs = [_msg("system", "S"), _msg("user", "u1"), _msg("assistant", "a1")]
        out = mem.compact(msgs)
        assert out == msgs

    def test_returns_list_copy_not_alias(self) -> None:
        mem = BufferMemory(max_messages=10)
        msgs = [_msg("user", "u1")]
        out = mem.compact(msgs)
        assert out is not msgs

    def test_long_history_trimmed_keeping_system(self) -> None:
        mem = BufferMemory(max_messages=4)
        msgs = [
            _msg("system", "S"),
            _msg("user", "u1"),
            _msg("assistant", "a1"),
            _msg("user", "u2"),
            _msg("assistant", "a2"),
            _msg("user", "u3"),
            _msg("assistant", "a3"),
            _msg("user", "u4"),
            _msg("assistant", "a4"),
        ]
        out = mem.compact(msgs)
        # System preserved, last 4 non-system kept.
        assert out[0].role == "system"
        assert [m.content for m in out[1:]] == ["u3", "a3", "u4", "a4"]

    def test_truncation_anchors_on_user_message(self) -> None:
        """If the tail starts with a `tool` message (orphaned after
        truncation), walk forward to the first `user`."""
        mem = BufferMemory(max_messages=3)
        msgs = [
            _msg("system", "S"),
            _msg("user", "u1"),
            _msg("assistant", "a1"),
            _msg("tool", "t1"),
            _msg("user", "u2"),
            _msg("assistant", "a2"),
        ]
        # max_messages=3 → tail of 3 = ['tool t1', 'user u2', 'assistant a2']
        # First role=user is at index 1 of the tail → drop t1.
        out = mem.compact(msgs)
        assert [m.role for m in out] == ["system", "user", "assistant"]
        assert [m.content for m in out] == ["S", "u2", "a2"]

    def test_no_user_in_tail_drops_tail_to_honour_hard_cap(self) -> None:
        """When the ``max_messages``-sized tail has no user message,
        the tail is dropped entirely. We do NOT walk backward past the
        cap — better a system-only context than a malformed conversation
        OR a soft cap that surprises the host."""
        mem = BufferMemory(max_messages=2)
        msgs = [
            _msg("system", "S"),
            _msg("user", "u1"),
            _msg("assistant", "a1"),
            _msg("tool", "t1"),
            _msg("tool", "t2"),
            _msg("assistant", "a2"),
        ]
        out = mem.compact(msgs)
        assert [m.role for m in out] == ["system"]
        # Crucially: the hard cap was honoured. No 5-message return.
        assert len([m for m in out if m.role != "system"]) == 0

    def test_max_messages_is_hard_cap(self) -> None:
        """No execution path should ever return more than ``max_messages``
        non-system messages."""
        mem = BufferMemory(max_messages=3)
        msgs: list[Message] = [_msg("system", "S")]
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append(_msg(role, f"{role}{i}"))
        out = mem.compact(msgs)
        non_system = [m for m in out if m.role != "system"]
        assert len(non_system) <= 3

    def test_short_history_with_orphan_tool_head_cleaned(self) -> None:
        """Even short histories must produce well-formed output. An
        input that begins with an orphan ``tool`` message gets the
        orphan stripped."""
        mem = BufferMemory(max_messages=20)
        msgs = [
            _msg("system", "S"),
            _msg("tool", "orphan"),
            _msg("user", "u1"),
            _msg("assistant", "a1"),
        ]
        out = mem.compact(msgs)
        # The orphan tool must be gone; first non-system is `user`.
        roles = [m.role for m in out]
        assert roles[0] == "system"
        assert roles[1] == "user"
        assert "tool" in [m.role for m in msgs]  # was in input
        # And the trailing well-formed turn is preserved.
        assert [m.content for m in out if m.role != "system"] == ["u1", "a1"]

    def test_no_user_anywhere_drops_all_non_system(self) -> None:
        """If no user message is reachable inside the cap, the tail is
        empty — system messages remain, others are dropped."""
        mem = BufferMemory(max_messages=5)
        msgs = [_msg("system", "S"), _msg("assistant", "a1"), _msg("tool", "t1")]
        out = mem.compact(msgs)
        assert [m.role for m in out] == ["system"]

    def test_multiple_system_messages_all_kept(self) -> None:
        mem = BufferMemory(max_messages=2)
        msgs = [
            _msg("system", "S1"),
            _msg("system", "S2"),
            _msg("user", "u1"),
            _msg("assistant", "a1"),
            _msg("user", "u2"),
            _msg("assistant", "a2"),
        ]
        out = mem.compact(msgs)
        assert [m.content for m in out if m.role == "system"] == ["S1", "S2"]
        assert [m.content for m in out if m.role != "system"] == ["u2", "a2"]

    def test_only_system_messages_unchanged(self) -> None:
        mem = BufferMemory(max_messages=1)
        msgs = [_msg("system", "S1"), _msg("system", "S2")]
        out = mem.compact(msgs)
        assert out == msgs


class TestBufferMemoryRecall:
    def test_returns_empty(self) -> None:
        assert BufferMemory().recall("anything") == []
        assert BufferMemory().recall("anything", k=100) == []


# ---------------------------------------------------------------------------
# Agent wiring — backward compat (memory=None unchanged)
# ---------------------------------------------------------------------------


class TestAgentMemoryNone:
    def test_default_agent_has_no_memory(self) -> None:
        provider = FakeLLMProvider([_text("done")])
        agent = Agent(provider)
        assert agent.memory is None
        assert agent.run("hi").output == "done"

    def test_explicit_none_runs_same(self) -> None:
        provider = FakeLLMProvider([_text("done")])
        agent = Agent(provider, memory=None)
        result = agent.run("hi")
        assert result.output == "done"


# ---------------------------------------------------------------------------
# Agent wiring — compact is called once before the loop
# ---------------------------------------------------------------------------


class _CountingMemory:
    """Records every compact() and recall() call for assertions."""

    def __init__(self, max_messages: int = 5) -> None:
        self.compact_calls = 0
        self.recall_calls: list[tuple[str, int]] = []
        self.last_compact_input: list[Message] | None = None
        self._inner = BufferMemory(max_messages=max_messages)

    def compact(self, messages: list[Message]) -> list[Message]:
        self.compact_calls += 1
        self.last_compact_input = list(messages)
        return self._inner.compact(messages)

    def recall(self, query: str, k: int = 5) -> list[Message]:
        self.recall_calls.append((query, k))
        return []


class TestAgentMemoryCompact:
    def test_compact_called_once_per_run(self) -> None:
        provider = FakeLLMProvider([_text("done")])
        memory = _CountingMemory()
        agent = Agent(provider, memory=memory)
        agent.run("hi")
        assert memory.compact_calls == 1

    def test_compact_called_with_initial_messages(self) -> None:
        provider = FakeLLMProvider([_text("done")])
        memory = _CountingMemory()
        agent = Agent(provider, memory=memory)
        agent.run("hi")
        assert memory.last_compact_input is not None
        assert any(m.role == "user" and m.content == "hi" for m in memory.last_compact_input)

    def test_compact_not_called_again_after_tool_call(self) -> None:
        """The compact happens ONCE before the loop. Tool calls inside
        the loop do not trigger additional compactions."""
        provider = FakeLLMProvider(
            [
                LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id="c1", name="noop", arguments={})],
                    model="fake",
                ),
                _text("done"),
            ]
        )
        memory = _CountingMemory()
        agent = Agent(provider, memory=memory)
        agent.registry.add(ToolSpec(name="noop", description="noop"), lambda: "ok")
        agent.run("hi")
        assert memory.compact_calls == 1

    def test_compact_reshaping_trims_input(self) -> None:
        """A long history is trimmed by the memory before going to the
        provider."""
        provider = FakeLLMProvider([_text("done")])
        agent = Agent(provider, memory=BufferMemory(max_messages=2))
        history = [
            Message(role="system", content="S"),
            Message(role="user", content="u1"),
            Message(role="assistant", content="a1"),
            Message(role="user", content="u2"),
            Message(role="assistant", content="a2"),
            Message(role="user", content="u3"),
        ]
        agent.run_messages(history)
        request = provider.calls[0]
        # The provider should have received the trimmed history.
        sent = request.messages
        assert len(sent) <= 3  # 1 system + 2 non-system
        assert sent[0].role == "system"

    def test_buggy_compact_isolated(self) -> None:
        class BadMemory:
            def compact(self, _msgs: list[Message]) -> list[Message]:
                raise RuntimeError("boom")

            def recall(self, _q: str, _k: int = 5) -> list[Message]:
                return []

        provider = FakeLLMProvider([_text("done")])
        agent = Agent(provider, memory=BadMemory())  # type: ignore[arg-type]
        # The agent must NOT propagate the memory error.
        result = agent.run("hi")
        assert result.output == "done"


# ---------------------------------------------------------------------------
# Agent wiring — register_recall_tool
# ---------------------------------------------------------------------------


class TestRegisterRecallTool:
    def test_registers_a_recall_tool_when_memory_configured(self) -> None:
        provider = FakeLLMProvider([_text("done")])
        agent = Agent(provider, memory=BufferMemory(max_messages=3))
        agent.register_recall_tool()
        assert "recall" in {spec.name for spec in agent.registry.specs()}

    def test_silent_noop_when_no_memory(self) -> None:
        provider = FakeLLMProvider([_text("done")])
        agent = Agent(provider, memory=None)
        # Must NOT raise; just silently skip.
        agent.register_recall_tool()
        assert "recall" not in {spec.name for spec in agent.registry.specs()}

    def test_recall_tool_returns_matches(self) -> None:
        class FakeMemory:
            def compact(self, messages: list[Message]) -> list[Message]:
                return messages

            def recall(self, query: str, k: int = 5) -> list[Message]:
                return [Message(role="user", content=f"past: {query}")]

        provider = FakeLLMProvider(
            [
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="r1",
                            name="recall",
                            arguments={"query": "what color was the button?"},
                        )
                    ],
                    model="fake",
                ),
                _text("found it"),
            ]
        )
        agent = Agent(provider, memory=FakeMemory())  # type: ignore[arg-type]
        agent.register_recall_tool()
        result = agent.run("retrieve")
        # The agent's history must contain the recall tool result.
        tool_msgs = [m for m in result.messages if m.role == "tool" and m.name == "recall"]
        assert tool_msgs
        assert "past: what color was the button?" in tool_msgs[0].content

    def test_recall_failure_is_reported_not_raised(self) -> None:
        class ExplosiveMemory:
            def compact(self, messages: list[Message]) -> list[Message]:
                return messages

            def recall(self, query: str, k: int = 5) -> list[Message]:
                raise RuntimeError("vector store offline")

        provider = FakeLLMProvider(
            [
                LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id="r1", name="recall", arguments={"query": "x"})],
                    model="fake",
                ),
                _text("ok"),
            ]
        )
        agent = Agent(provider, memory=ExplosiveMemory())  # type: ignore[arg-type]
        agent.register_recall_tool()
        result = agent.run("retrieve")
        tool_msgs = [m for m in result.messages if m.role == "tool" and m.name == "recall"]
        # Tool succeeded structurally; the error is reported in the result.
        assert tool_msgs
        assert "vector store offline" in tool_msgs[0].content

    def test_custom_name_honored(self) -> None:
        provider = FakeLLMProvider([_text("done")])
        agent = Agent(provider, memory=BufferMemory(max_messages=3))
        agent.register_recall_tool(name="search_past")
        names = {spec.name for spec in agent.registry.specs()}
        assert "search_past" in names
        assert "recall" not in names


# ---------------------------------------------------------------------------
# Sanity — Memory does NOT break post_turn_hook, cancel_token, or trace
# ---------------------------------------------------------------------------


class TestMemoryComposesWithExistingFeatures:
    def test_with_post_turn_hook(self) -> None:
        provider = FakeLLMProvider([_text("first"), _text("second")])

        def hook(_ctx: Any) -> Any:
            return Message(role="user", content="try again")

        agent = Agent(
            provider,
            memory=BufferMemory(max_messages=20),
            post_turn_hook=hook,
            max_corrections_per_run=1,
        )
        result = agent.run("hi")
        # Two iterations happened: first response + correction + second.
        assert result.output == "second"

    def test_with_trace_emitter(self) -> None:
        from autoagent.trace import TraceEmitter

        events: list[Any] = []
        provider = FakeLLMProvider([_text("done")])
        agent = Agent(
            provider,
            memory=BufferMemory(max_messages=20),
            trace=TraceEmitter(on_event=events.append),
        )
        agent.run("hi")
        types = [e.type for e in events]
        assert "run_start" in types
        assert "run_end" in types
