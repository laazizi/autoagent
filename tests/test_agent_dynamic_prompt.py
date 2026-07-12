"""Tests for the ``system_prompt`` callable feature (added in 0.7.0).

Public contract:

* ``Agent(system_prompt="...")`` continues to work as before — string
  passes through unchanged.
* ``Agent(system_prompt=lambda: "...")`` invokes the callable at the
  start of every ``run()`` call. Each run sees a fresh value.
* ``Agent.render_system_prompt()`` returns the resolved string and can
  be called any time by hosts that persist conversations across HTTP
  requests (to refresh stored system messages between turns).
* A callable that raises is caught, logged, and the run falls back to
  ``DEFAULT_SYSTEM_PROMPT`` rather than crashing.
* A callable that returns ``None`` is treated as the default prompt.
"""

from __future__ import annotations

from autoagent.agent import DEFAULT_SYSTEM_PROMPT, Agent

from .conftest import FakeLLMProvider


class TestStaticString:
    def test_default_is_default(self) -> None:
        provider = FakeLLMProvider(responses=["ok"])
        agent = Agent(provider=provider)
        assert agent.render_system_prompt() == DEFAULT_SYSTEM_PROMPT

    def test_custom_string_passes_through(self) -> None:
        provider = FakeLLMProvider(responses=["ok"])
        agent = Agent(provider=provider, system_prompt="custom prompt")
        assert agent.render_system_prompt() == "custom prompt"

    def test_run_uses_custom_string(self) -> None:
        provider = FakeLLMProvider(responses=["ok"])
        agent = Agent(provider=provider, system_prompt="be terse")
        agent.run("hi")
        assert provider.calls[0].messages[0].role == "system"
        assert provider.calls[0].messages[0].content == "be terse"


class TestCallable:
    def test_callable_invoked_on_render(self) -> None:
        provider = FakeLLMProvider(responses=["ok"])
        agent = Agent(provider=provider, system_prompt=lambda: "live prompt")
        assert agent.render_system_prompt() == "live prompt"

    def test_callable_invoked_each_run(self) -> None:
        provider = FakeLLMProvider(responses=["ok", "ok"])
        counter = {"n": 0}

        def build() -> str:
            counter["n"] += 1
            return f"prompt #{counter['n']}"

        agent = Agent(provider=provider, system_prompt=build)
        agent.run("hi")
        agent.run("hi again")
        # Two runs => callable called twice, each saw a fresh prompt.
        assert counter["n"] == 2
        assert provider.calls[0].messages[0].content == "prompt #1"
        assert provider.calls[1].messages[0].content == "prompt #2"

    def test_callable_returning_none_falls_back(self) -> None:
        provider = FakeLLMProvider(responses=["ok"])
        agent = Agent(provider=provider, system_prompt=lambda: None)  # type: ignore[arg-type,return-value]
        assert agent.render_system_prompt() == DEFAULT_SYSTEM_PROMPT

    def test_callable_returning_non_string_is_coerced(self) -> None:
        provider = FakeLLMProvider(responses=["ok"])
        agent = Agent(provider=provider, system_prompt=lambda: 42)  # type: ignore[arg-type,return-value]
        # Non-string return is coerced via str() — covers numeric counters,
        # template objects, etc. Hosts shouldn't rely on this but we don't
        # want to crash either.
        assert agent.render_system_prompt() == "42"

    def test_callable_raising_falls_back_to_default(self) -> None:
        provider = FakeLLMProvider(responses=["ok"])

        def broken() -> str:
            raise RuntimeError("boom")

        agent = Agent(provider=provider, system_prompt=broken)
        # Must not raise — falls back to default and the run proceeds.
        assert agent.render_system_prompt() == DEFAULT_SYSTEM_PROMPT
        result = agent.run("hi")
        assert result.output == "ok"
        assert provider.calls[0].messages[0].content == DEFAULT_SYSTEM_PROMPT


class TestRefreshPattern:
    """The canonical FastAPI / chat-session pattern.

    Hosts that persist history across HTTP requests should replace the
    stale system message in their stored history with a fresh one on
    every turn. ``render_system_prompt()`` is the supported way to get
    that fresh string.
    """

    def test_refresh_pattern_works(self) -> None:
        provider = FakeLLMProvider(responses=["ok"])
        state = {"step": 0}

        def build() -> str:
            return f"step={state['step']}"

        agent = Agent(provider=provider, system_prompt=build)

        # Tour 1
        state["step"] = 1
        prompt_t1 = agent.render_system_prompt()
        assert prompt_t1 == "step=1"

        # Tour 2 — state changed between turns, prompt reflects it
        state["step"] = 2
        prompt_t2 = agent.render_system_prompt()
        assert prompt_t2 == "step=2"
