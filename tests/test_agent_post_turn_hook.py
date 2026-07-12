"""Tests for the post_turn_hook feature added in 0.2.0.

The hook lets a host plug verification logic between the LLM emitting a
final text response and the agent returning to the caller. It can
optionally inject a correction message that causes the agent to take
another turn (e.g. "your code crashes — fix it").
"""

from __future__ import annotations

from autoagent.agent import Agent, AgentTurnContext
from autoagent.schema import LLMResponse, Message, ToolCall, ToolSpec

from .conftest import FakeLLMProvider


def _text(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="fake")


def _tool(calls: list[ToolCall], content: str = "") -> LLMResponse:
    return LLMResponse(content=content, tool_calls=calls, model="fake")


# ---------------------------------------------------------------------------
# Hook returns None — normal end of turn
# ---------------------------------------------------------------------------


class TestHookReturnsNone:
    def test_run_completes_when_hook_returns_none(self) -> None:
        provider = FakeLLMProvider([_text("done")])

        def hook(ctx: AgentTurnContext) -> Message | None:
            return None

        agent = Agent(provider, post_turn_hook=hook)
        result = agent.run("hi")
        assert result.output == "done"

    def test_hook_receives_context_with_correct_count_zero(self) -> None:
        provider = FakeLLMProvider([_text("done")])
        seen: list[AgentTurnContext] = []

        def hook(ctx: AgentTurnContext) -> Message | None:
            seen.append(ctx)
            return None

        agent = Agent(provider, post_turn_hook=hook)
        agent.run("hi")
        assert len(seen) == 1
        assert seen[0].correction_count == 0
        # new_messages should include at least the assistant final response
        assert any(m.role == "assistant" for m in seen[0].new_messages)

    def test_hook_sees_tool_calls_made_during_turn(self) -> None:
        provider = FakeLLMProvider(
            [
                _tool([ToolCall(id="c1", name="echo", arguments={"x": "hi"})]),
                _text("ok"),
            ]
        )

        def echo(x: str) -> str:
            return x

        agent = Agent(provider)
        agent.registry.add(ToolSpec(name="echo", description="echo"), echo)

        seen: list[AgentTurnContext] = []

        def hook(ctx: AgentTurnContext) -> Message | None:
            seen.append(ctx)
            return None

        agent.post_turn_hook = hook
        agent.run("hi")

        assert len(seen) == 1
        names = [c.name for c in seen[0].tool_calls]
        assert "echo" in names


# ---------------------------------------------------------------------------
# Hook returns a correction Message — agent gets another turn
# ---------------------------------------------------------------------------


class TestHookReturnsCorrection:
    def test_correction_triggers_another_turn(self) -> None:
        # First LLM response: "done" (would normally end).
        # Hook injects a correction.
        # Second LLM response: "fixed".
        provider = FakeLLMProvider([_text("done"), _text("fixed")])

        calls = {"n": 0}

        def hook(ctx: AgentTurnContext) -> Message | None:
            calls["n"] += 1
            if ctx.correction_count == 0:
                return Message(role="user", content="actually no, please redo")
            return None

        agent = Agent(provider, post_turn_hook=hook, max_corrections_per_run=2)
        result = agent.run("hi")
        assert result.output == "fixed"
        assert calls["n"] == 2  # first time correction injected, second time None

    def test_correction_message_appears_in_history(self) -> None:
        provider = FakeLLMProvider([_text("done"), _text("fixed")])

        def hook(ctx: AgentTurnContext) -> Message | None:
            if ctx.correction_count == 0:
                return Message(role="user", content="ERROR: try again")
            return None

        agent = Agent(provider, post_turn_hook=hook, max_corrections_per_run=2)
        result = agent.run("hi")
        contents = [m.content for m in result.messages]
        assert any("ERROR: try again" in c for c in contents)

    def test_correction_count_increments(self) -> None:
        provider = FakeLLMProvider([_text("v1"), _text("v2"), _text("v3")])
        seen_counts: list[int] = []

        def hook(ctx: AgentTurnContext) -> Message | None:
            seen_counts.append(ctx.correction_count)
            if ctx.correction_count < 2:
                return Message(role="user", content="again")
            return None

        agent = Agent(provider, post_turn_hook=hook, max_corrections_per_run=5)
        agent.run("hi")
        assert seen_counts == [0, 1, 2]


# ---------------------------------------------------------------------------
# Correction budget (max_corrections_per_run)
# ---------------------------------------------------------------------------


class TestCorrectionBudget:
    def test_budget_zero_disables_hook(self) -> None:
        provider = FakeLLMProvider([_text("done")])
        called = {"n": 0}

        def hook(ctx: AgentTurnContext) -> Message | None:
            called["n"] += 1
            return Message(role="user", content="redo")

        agent = Agent(provider, post_turn_hook=hook, max_corrections_per_run=0)
        result = agent.run("hi")
        assert result.output == "done"
        assert called["n"] == 0, "Hook must not be called when budget is zero"

    def test_budget_caps_corrections(self) -> None:
        # Hook always tries to correct; budget=1 should allow exactly 1
        # correction, so total LLM calls = 2 (initial + 1 correction).
        provider = FakeLLMProvider([_text("v1"), _text("v2"), _text("v3")])

        def hook(ctx: AgentTurnContext) -> Message | None:
            return Message(role="user", content="redo")

        agent = Agent(provider, post_turn_hook=hook, max_corrections_per_run=1)
        result = agent.run("hi")
        assert result.output == "v2"
        # Provider should have been called exactly 2 times.
        assert len(provider.calls) == 2


# ---------------------------------------------------------------------------
# Hook exceptions are isolated
# ---------------------------------------------------------------------------


class TestHookExceptions:
    def test_hook_exception_does_not_break_run(self) -> None:
        provider = FakeLLMProvider([_text("done")])

        def hook(ctx: AgentTurnContext) -> Message | None:
            raise RuntimeError("hook is broken")

        agent = Agent(provider, post_turn_hook=hook)
        # The run must complete normally despite the hook raising.
        result = agent.run("hi")
        assert result.output == "done"


# ---------------------------------------------------------------------------
# No-hook path (default behaviour unchanged)
# ---------------------------------------------------------------------------


class TestNoHook:
    def test_default_behaviour_when_hook_is_none(self) -> None:
        provider = FakeLLMProvider([_text("done")])
        agent = Agent(provider)  # no post_turn_hook
        result = agent.run("hi")
        assert result.output == "done"

    def test_hook_attribute_defaults_to_none(self) -> None:
        agent = Agent(FakeLLMProvider([_text("done")]))
        assert agent.post_turn_hook is None
        assert agent.max_corrections_per_run == 1
