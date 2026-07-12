"""Tests for `autoagent.trace.TraceEmitter` and its wiring into `Agent` (0.5.0).

The public contract:

* `TraceEmitter(file=..., on_event=...)` lets a host observe an agent
  run as a stream of typed events without changing the agent's
  behaviour.
* When `Agent(trace=None)` (the default), the agent emits nothing and
  pays no overhead — zero behaviour change vs prior versions.
* File and callback both work; both together work; neither is fine
  (no-op).
* The emitter never propagates an error: a misbehaving callback or a
  full disk must not break the agent loop.
* Events form a tree via `parent_id` so a UI can group tool calls
  under their owning LLM step.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from autoagent.agent import Agent
from autoagent.errors import AgentCancelled, MaxStepsExceeded
from autoagent.schema import LLMResponse, Message, ToolCall, ToolSpec
from autoagent.trace import TraceEmitter, TraceEvent, truncate_preview

from .conftest import FakeLLMProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="fake")


def _tool(calls: list[ToolCall], content: str = "") -> LLMResponse:
    return LLMResponse(content=content, tool_calls=calls, model="fake")


class _Collector:
    """Drop-in `on_event` callback that records every event it receives."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def __call__(self, event: TraceEvent) -> None:
        self.events.append(event)

    def types(self) -> list[str]:
        return [e.type for e in self.events]

    def by_type(self, type_: str) -> list[TraceEvent]:
        return [e for e in self.events if e.type == type_]


# ---------------------------------------------------------------------------
# TraceEmitter standalone behaviour
# ---------------------------------------------------------------------------


class TestEmitterBasics:
    def test_no_file_no_callback_is_legal_noop(self) -> None:
        emitter = TraceEmitter()
        span_id = emitter.emit("custom", {"x": 1})
        assert isinstance(span_id, str)
        assert len(span_id) == 16  # 8 bytes hex
        emitter.close()  # idempotent

    def test_callback_receives_event(self) -> None:
        collector = _Collector()
        emitter = TraceEmitter(on_event=collector)
        emitter.emit("foo", {"k": "v"})
        assert len(collector.events) == 1
        assert collector.events[0].type == "foo"
        assert collector.events[0].payload == {"k": "v"}

    def test_callback_exception_isolated(self) -> None:
        def boom(_: TraceEvent) -> None:
            raise RuntimeError("callback broke")

        emitter = TraceEmitter(on_event=boom)
        # Must NOT raise even though the callback always fails.
        sid = emitter.emit("x", {})
        assert isinstance(sid, str)

    def test_parent_child_chain(self) -> None:
        collector = _Collector()
        emitter = TraceEmitter(on_event=collector)
        root = emitter.emit("root", {})
        child = emitter.emit("child", {}, parent_id=root)
        grand = emitter.emit("grand", {}, parent_id=child)
        assert collector.events[0].parent_id is None
        assert collector.events[1].parent_id == root
        assert collector.events[2].parent_id == child
        assert grand != child != root


# ---------------------------------------------------------------------------
# JSONL file output
# ---------------------------------------------------------------------------


class TestEmitterFile:
    def test_appends_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        emitter = TraceEmitter(file=path)
        emitter.emit("a", {"i": 1})
        emitter.emit("b", {"i": 2})
        emitter.close()

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        record0 = json.loads(lines[0])
        record1 = json.loads(lines[1])
        assert record0["type"] == "a"
        assert record0["payload"] == {"i": 1}
        assert record1["type"] == "b"
        assert "span_id" in record0
        assert "ts" in record0

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        emitter = TraceEmitter(file=tmp_path / "t.jsonl")
        emitter.close()
        emitter.close()  # second close must not raise

    def test_context_manager_closes_file(self, tmp_path: Path) -> None:
        path = tmp_path / "ctx.jsonl"
        with TraceEmitter(file=path) as emitter:
            emitter.emit("hi", {})
        # After exit the file is closed; another open should see the line.
        assert path.read_text(encoding="utf-8").strip() != ""

    def test_existing_file_is_appended_not_truncated(self, tmp_path: Path) -> None:
        path = tmp_path / "append.jsonl"
        path.write_text('{"type": "preexisting"}\n', encoding="utf-8")
        emitter = TraceEmitter(file=path)
        emitter.emit("new", {})
        emitter.close()
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["type"] == "preexisting"
        assert json.loads(lines[1])["type"] == "new"

    def test_external_file_handle_not_closed(self, tmp_path: Path) -> None:
        path = tmp_path / "ext.jsonl"
        handle = path.open("a", encoding="utf-8")
        try:
            emitter = TraceEmitter(file=handle)
            emitter.emit("x", {})
            emitter.close()
            assert not handle.closed  # host owns the handle
            handle.write("after-close-on-emitter\n")
        finally:
            handle.close()


# ---------------------------------------------------------------------------
# truncate_preview helper
# ---------------------------------------------------------------------------


class TestTruncatePreview:
    def test_none_renders_empty(self) -> None:
        assert truncate_preview(None) == ""

    def test_short_string_unchanged(self) -> None:
        assert truncate_preview("hello") == "hello"

    def test_long_string_truncated(self) -> None:
        big = "x" * 1000
        out = truncate_preview(big)
        assert out.endswith("...")
        assert len(out) <= 250  # limit + ellipsis

    def test_dict_jsonified(self) -> None:
        assert truncate_preview({"a": 1}) == '{"a": 1}'

    def test_redacts_bearer_token_in_string(self) -> None:
        out = truncate_preview("Authorization: Bearer sk-abcdef1234567890")
        assert "sk-abcdef1234567890" not in out
        assert "REDACTED" in out

    def test_redacts_api_key_in_dict(self) -> None:
        out = truncate_preview({"api_key": "sk-abcdef1234567890"})
        assert "sk-abcdef1234567890" not in out
        assert "REDACTED" in out

    def test_redacts_x_api_key_in_header(self) -> None:
        out = truncate_preview("x-api-key: sk-secret-abcdef1234")
        assert "sk-secret-abcdef1234" not in out

    def test_redacts_url_key_param(self) -> None:
        out = truncate_preview("https://api.example.com/v1?key=SUPER-SECRET-KEY")
        assert "SUPER-SECRET-KEY" not in out


# ---------------------------------------------------------------------------
# Secret redaction reaches all agent emit sites
# ---------------------------------------------------------------------------


class TestAgentTraceSecretRedaction:
    def test_tool_arguments_redacted(self) -> None:
        collector = _Collector()
        provider = FakeLLMProvider(
            [
                _tool(
                    [
                        ToolCall(
                            id="c1",
                            name="echo",
                            arguments={"api_key": "sk-leak-abcdef1234"},
                        )
                    ]
                ),
                _text("done"),
            ]
        )
        agent = Agent(provider, trace=TraceEmitter(on_event=collector))
        agent.registry.add(ToolSpec(name="echo", description="echo"), lambda **_: "ok")
        agent.run("hi")
        start = collector.by_type("tool_call_start")[0]
        assert "sk-leak-abcdef1234" not in start.payload["arguments_preview"]
        assert "REDACTED" in start.payload["arguments_preview"]

    def test_tool_error_message_redacted(self) -> None:
        """A formal credential form embedded in a tool exception message
        is redacted in `tool_call_end.content_preview`.

        We use the `Authorization: Bearer ...` shape because the lib's
        `SecretRedactingFilter` only matches formal patterns (header,
        JSON / dict field, ``?key=``); free prose like "the API key
        is sk-..." is intentionally NOT caught to avoid redacting
        innocent text. Hosts that need prose-level redaction should
        layer their own filter on top of `on_event`.
        """
        collector = _Collector()
        provider = FakeLLMProvider(
            [
                _tool([ToolCall(id="c1", name="boom", arguments={})]),
                _text("done"),
            ]
        )

        def boom() -> str:
            raise ValueError("Upstream rejected Authorization: Bearer sk-secret-leak-1234567890")

        agent = Agent(provider, trace=TraceEmitter(on_event=collector))
        agent.registry.add(ToolSpec(name="boom", description="boom"), boom)
        agent.run("hi")
        end = collector.by_type("tool_call_end")[0]
        assert "sk-secret-leak-1234567890" not in end.payload["content_preview"]

    def test_llm_response_content_redacted(self) -> None:
        collector = _Collector()
        provider = FakeLLMProvider([_text("The token is Bearer sk-leak-xyz-abcdef1234567890")])
        agent = Agent(provider, trace=TraceEmitter(on_event=collector))
        agent.run("hi")
        resp = collector.by_type("llm_response")[0]
        end = collector.by_type("run_end")[0]
        assert "sk-leak-xyz-abcdef1234567890" not in resp.payload["content_preview"]
        assert "sk-leak-xyz-abcdef1234567890" not in end.payload["output_preview"]

    def test_post_turn_hook_correction_redacted(self) -> None:
        collector = _Collector()
        provider = FakeLLMProvider([_text("first"), _text("second")])

        def hook(_ctx: Any) -> Any:
            return Message(
                role="user",
                content="please use api_key=sk-injected-leak-1234567890",
            )

        agent = Agent(
            provider,
            trace=TraceEmitter(on_event=collector),
            post_turn_hook=hook,
            max_corrections_per_run=1,
        )
        agent.run("hi")
        corr = collector.by_type("post_turn_hook_correction")[0]
        assert "sk-injected-leak-1234567890" not in corr.payload["content_preview"]


# ---------------------------------------------------------------------------
# Hardening — emit() never raises, even with a broken user-supplied clock
# ---------------------------------------------------------------------------


class TestEmitterHardening:
    def test_clock_that_raises_falls_back_silently(self) -> None:
        def bad_clock() -> float:
            raise RuntimeError("clock broke")

        emitter = TraceEmitter(clock=bad_clock)
        # Must not raise.
        sid = emitter.emit("x", {"k": 1})
        assert isinstance(sid, str)
        assert len(sid) == 16

    def test_agent_with_broken_clock_emitter_still_runs(self) -> None:
        provider = FakeLLMProvider([_text("done")])

        def bad_clock() -> float:
            raise RuntimeError("clock broke")

        emitter = TraceEmitter(clock=bad_clock)
        agent = Agent(provider, trace=emitter)
        result = agent.run("hi")
        assert result.output == "done"

    def test_concurrent_emits_appear_in_order(self, tmp_path: Path) -> None:
        """span_id and ts are generated INSIDE the lock so the JSONL
        file order matches generation order. Without the fix, two
        threads could swap timestamps."""
        path = tmp_path / "concurrent.jsonl"
        emitter = TraceEmitter(file=path)
        n_threads = 8
        per_thread = 25
        barrier = threading.Barrier(n_threads)

        def worker(tid: int) -> None:
            barrier.wait()
            for i in range(per_thread):
                emitter.emit(f"t{tid}-{i}", {"tid": tid, "i": i})

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        emitter.close()

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == n_threads * per_thread
        events = [json.loads(line) for line in lines]
        # Timestamps must be non-decreasing in file order — the lock
        # guarantees this only if ts is captured under the lock.
        timestamps = [e["ts"] for e in events]
        assert timestamps == sorted(
            timestamps
        ), "events out of timestamp order — span_id/ts must be generated under the lock"


# ---------------------------------------------------------------------------
# Agent wiring — events emitted on a normal run
# ---------------------------------------------------------------------------


class TestAgentTraceWiring:
    def test_default_agent_has_no_trace(self) -> None:
        provider = FakeLLMProvider([_text("done")])
        agent = Agent(provider)
        assert agent.trace is None
        # And the run still works fine.
        result = agent.run("hi")
        assert result.output == "done"

    def test_simple_run_emits_run_start_request_response_run_end(self) -> None:
        collector = _Collector()
        provider = FakeLLMProvider([_text("done")])
        agent = Agent(provider, trace=TraceEmitter(on_event=collector))
        agent.run("hi")
        assert collector.types() == [
            "run_start",
            "llm_request",
            "llm_response",
            "run_end",
        ]
        end = collector.by_type("run_end")[0]
        assert end.payload["status"] == "ok"
        assert end.payload["steps"] == 1
        assert end.payload["output_preview"] == "done"

    def test_tool_call_emits_start_and_end(self) -> None:
        collector = _Collector()
        provider = FakeLLMProvider(
            [
                _tool([ToolCall(id="c1", name="echo", arguments={"x": 1})]),
                _text("done"),
            ]
        )
        agent = Agent(provider, trace=TraceEmitter(on_event=collector))
        agent.registry.add(ToolSpec(name="echo", description="echo"), lambda x: x)
        agent.run("hi")
        types = collector.types()
        assert "tool_call_start" in types
        assert "tool_call_end" in types
        start = collector.by_type("tool_call_start")[0]
        end = collector.by_type("tool_call_end")[0]
        assert start.payload["name"] == "echo"
        assert start.payload["call_id"] == "c1"
        assert end.payload["name"] == "echo"
        assert end.payload["status"] == "ok"
        assert isinstance(end.payload["duration_ms"], int)
        assert end.payload["duration_ms"] >= 0

    def test_tool_failure_marks_status_error(self) -> None:
        collector = _Collector()
        provider = FakeLLMProvider(
            [
                _tool([ToolCall(id="c1", name="boom", arguments={})]),
                _text("done"),
            ]
        )

        def boom() -> str:
            raise RuntimeError("kaboom")

        agent = Agent(provider, trace=TraceEmitter(on_event=collector))
        agent.registry.add(ToolSpec(name="boom", description="boom"), boom)
        agent.run("hi")
        end = collector.by_type("tool_call_end")[0]
        assert end.payload["status"] == "error"
        assert "kaboom" in end.payload["content_preview"]

    def test_events_form_a_tree(self) -> None:
        collector = _Collector()
        provider = FakeLLMProvider(
            [
                _tool([ToolCall(id="c1", name="noop", arguments={})]),
                _text("done"),
            ]
        )
        agent = Agent(provider, trace=TraceEmitter(on_event=collector))
        agent.registry.add(ToolSpec(name="noop", description="noop"), lambda: "ok")
        agent.run("hi")

        run_start = collector.by_type("run_start")[0]
        first_request = collector.by_type("llm_request")[0]
        first_response = collector.by_type("llm_response")[0]
        tool_start = collector.by_type("tool_call_start")[0]
        tool_end = collector.by_type("tool_call_end")[0]
        run_end = collector.by_type("run_end")[0]

        assert run_start.parent_id is None
        assert first_request.parent_id == run_start.span_id
        assert first_response.parent_id == first_request.span_id
        assert tool_start.parent_id == first_request.span_id
        assert tool_end.parent_id == tool_start.span_id
        assert run_end.parent_id == run_start.span_id


# ---------------------------------------------------------------------------
# Agent wiring — error / cancellation paths
# ---------------------------------------------------------------------------


class TestAgentTraceErrorPaths:
    def test_cancelled_emits_cancelled_and_run_end(self) -> None:
        collector = _Collector()
        provider = FakeLLMProvider([_text("should-not-be-reached")])
        token = threading.Event()
        token.set()
        agent = Agent(provider, trace=TraceEmitter(on_event=collector))

        with pytest.raises(AgentCancelled):
            agent.run("hi", cancel_token=token)

        types = collector.types()
        assert "cancelled" in types
        end = collector.by_type("run_end")[0]
        assert end.payload["status"] == "cancelled"

    def test_max_steps_emits_max_steps_exceeded(self) -> None:
        collector = _Collector()
        # The provider keeps emitting tool calls forever — agent runs out
        # of steps.
        infinite_tool = _tool([ToolCall(id="c", name="noop", arguments={})])
        provider = FakeLLMProvider([infinite_tool] * 20)
        agent = Agent(
            provider,
            trace=TraceEmitter(on_event=collector),
            max_steps=2,
        )
        agent.registry.add(ToolSpec(name="noop", description="noop"), lambda: "ok")
        with pytest.raises(MaxStepsExceeded):
            agent.run("hi")
        assert "max_steps_exceeded" in collector.types()
        end = collector.by_type("run_end")[0]
        assert end.payload["status"] == "max_steps"


# ---------------------------------------------------------------------------
# Agent wiring — post_turn_hook events
# ---------------------------------------------------------------------------


class TestAgentTracePostTurnHook:
    def test_hook_invoked_event(self) -> None:
        collector = _Collector()
        provider = FakeLLMProvider([_text("first"), _text("second")])

        def hook(_ctx: Any) -> Any:
            return Message(role="user", content="please retry")

        agent = Agent(
            provider,
            trace=TraceEmitter(on_event=collector),
            post_turn_hook=hook,
            max_corrections_per_run=1,
        )
        agent.run("hi")

        assert "post_turn_hook_invoked" in collector.types()
        assert "post_turn_hook_correction" in collector.types()

    def test_hook_returning_none_emits_invoked_only(self) -> None:
        collector = _Collector()
        provider = FakeLLMProvider([_text("done")])

        agent = Agent(
            provider,
            trace=TraceEmitter(on_event=collector),
            post_turn_hook=lambda _ctx: None,
        )
        agent.run("hi")
        assert "post_turn_hook_invoked" in collector.types()
        assert "post_turn_hook_correction" not in collector.types()


# ---------------------------------------------------------------------------
# Failure isolation: a broken emitter must not break the agent
# ---------------------------------------------------------------------------


class TestAgentTraceFailureIsolation:
    def test_callback_that_raises_does_not_break_run(self) -> None:
        provider = FakeLLMProvider([_text("done")])

        def broken_callback(_: TraceEvent) -> None:
            raise RuntimeError("ouch")

        agent = Agent(provider, trace=TraceEmitter(on_event=broken_callback))
        # Even with a perpetually-raising callback the run completes.
        result = agent.run("hi")
        assert result.output == "done"

    def test_broken_emitter_itself_is_isolated(self) -> None:
        """If the TraceEmitter object itself raises (e.g. host monkey-
        patched it), the agent guards a second time and the run still
        succeeds."""
        provider = FakeLLMProvider([_text("done")])

        class BrokenEmitter:
            def emit(self, *_a: Any, **_kw: Any) -> str:
                raise RuntimeError("emitter dead")

        agent = Agent(provider, trace=BrokenEmitter())  # type: ignore[arg-type]
        result = agent.run("hi")
        assert result.output == "done"


# ---------------------------------------------------------------------------
# JSONL durability (the canonical persistence path used by hosts)
# ---------------------------------------------------------------------------


class TestAgentTraceFileEndToEnd:
    def test_full_run_persists_typed_events(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        provider = FakeLLMProvider(
            [
                _tool([ToolCall(id="c1", name="noop", arguments={})]),
                _text("done"),
            ]
        )
        with TraceEmitter(file=path) as trace:
            agent = Agent(provider, trace=trace)
            agent.registry.add(ToolSpec(name="noop", description="noop"), lambda: "ok")
            agent.run("hi")

        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        types = [e["type"] for e in events]
        assert types[0] == "run_start"
        assert types[-1] == "run_end"
        assert "tool_call_start" in types
        assert "tool_call_end" in types
        # Every event has the canonical fields.
        for e in events:
            assert set(e.keys()) >= {"type", "span_id", "parent_id", "ts", "payload"}
