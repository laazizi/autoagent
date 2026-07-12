"""Tests for cooperative cancellation in `agent.run_messages` (added in 0.3.0).

The lib accepts a `cancel_token: threading.Event` on `run` / `run_messages`.
When the host sets the event mid-flight, the agent loop checks the token at
the start of every iteration and raises `AgentCancelled` instead of
issuing the next LLM call. The exception propagates to the caller, which
typically catches it and reports cancellation in the UI.

We do NOT promise to interrupt an HTTP call already in flight — only to
stop scheduling new ones.
"""

from __future__ import annotations

import threading

import pytest

from autoagent.agent import Agent
from autoagent.errors import AgentCancelled
from autoagent.schema import LLMResponse, Message, ToolCall, ToolSpec

from .conftest import FakeLLMProvider


def _text(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="fake")


def _tool(calls: list[ToolCall], content: str = "") -> LLMResponse:
    return LLMResponse(content=content, tool_calls=calls, model="fake")


# ---------------------------------------------------------------------------
# Default behaviour unchanged
# ---------------------------------------------------------------------------


class TestCancelTokenNone:
    def test_no_token_runs_normally(self) -> None:
        provider = FakeLLMProvider([_text("done")])
        agent = Agent(provider)
        result = agent.run("hi")
        assert result.output == "done"

    def test_token_unset_runs_normally(self) -> None:
        provider = FakeLLMProvider([_text("done")])
        token = threading.Event()
        agent = Agent(provider)
        result = agent.run("hi", cancel_token=token)
        assert result.output == "done"


# ---------------------------------------------------------------------------
# Cancel before any LLM call
# ---------------------------------------------------------------------------


class TestCancelBeforeStart:
    def test_token_pre_set_raises_without_calling_provider(self) -> None:
        provider = FakeLLMProvider([_text("should-not-be-reached")])
        token = threading.Event()
        token.set()  # set BEFORE running

        agent = Agent(provider)
        with pytest.raises(AgentCancelled):
            agent.run("hi", cancel_token=token)
        # Critical: no API call must have been made.
        assert len(provider.calls) == 0


# ---------------------------------------------------------------------------
# Cancel mid-loop (between iterations)
# ---------------------------------------------------------------------------


class TestCancelMidRun:
    def test_token_set_after_first_iteration_stops_loop(self) -> None:
        """After the first LLM response (which triggers a tool call), the
        token is set. The next iteration must raise before calling the
        provider again."""
        provider = FakeLLMProvider(
            [
                _tool([ToolCall(id="c1", name="noop", arguments={})]),
                _text("should-not-be-reached"),
            ]
        )

        token = threading.Event()

        def noop() -> str:
            token.set()  # cancel right after the first tool runs
            return "ok"

        agent = Agent(provider)
        agent.registry.add(ToolSpec(name="noop", description="noop"), noop)

        with pytest.raises(AgentCancelled):
            agent.run("hi", cancel_token=token)
        # Only the first LLM call should have happened — the second one
        # (after the tool returned) must be skipped.
        assert len(provider.calls) == 1


# ---------------------------------------------------------------------------
# Cancel via run() vs run_messages()
# ---------------------------------------------------------------------------


class TestCancelEntryPoints:
    def test_run_messages_accepts_cancel_token(self) -> None:
        provider = FakeLLMProvider([_text("done")])
        agent = Agent(provider)
        token = threading.Event()
        token.set()
        with pytest.raises(AgentCancelled):
            agent.run_messages(
                [Message(role="system", content="s"), Message(role="user", content="u")],
                cancel_token=token,
            )

    def test_run_forwards_cancel_token_to_run_messages(self) -> None:
        provider = FakeLLMProvider([_text("done")])
        agent = Agent(provider)
        token = threading.Event()
        token.set()
        with pytest.raises(AgentCancelled):
            agent.run("hi", cancel_token=token)


# ---------------------------------------------------------------------------
# AgentCancelled inherits from AutoAgentError
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_agent_cancelled_is_autoagent_error(self) -> None:
        from autoagent.errors import AutoAgentError

        assert issubclass(AgentCancelled, AutoAgentError)

    def test_caller_can_catch_via_base(self) -> None:
        from autoagent.errors import AutoAgentError

        provider = FakeLLMProvider([_text("done")])
        token = threading.Event()
        token.set()
        agent = Agent(provider)

        try:
            agent.run("hi", cancel_token=token)
        except AutoAgentError:
            return
        raise AssertionError("AgentCancelled was not caught by AutoAgentError")
