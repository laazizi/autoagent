"""RunState / checkpoint / Agent.resume — long-running agents (0.11.0)."""

from __future__ import annotations

import json
import threading

import pytest

from autoagent.agent import Agent, RunState
from autoagent.errors import AgentCancelled, MaxStepsExceeded, TokenBudgetExceeded
from autoagent.schema import LLMResponse, Message, TokenUsage, ToolCall

from .conftest import FakeLLMProvider


def _text(content: str, usage: TokenUsage | None = None) -> LLMResponse:
    return LLMResponse(content=content, model="fake", usage=usage)


def _tool(name: str, call_id: str, usage: TokenUsage | None = None) -> LLMResponse:
    return LLMResponse(
        content="", model="fake",
        tool_calls=[ToolCall(id=call_id, name=name, arguments={})],
        usage=usage,
    )


def _agent(responses, **kwargs) -> Agent:
    agent = Agent(FakeLLMProvider(responses), **kwargs)

    @agent.tool
    def compter() -> dict:
        """Compte quelque chose."""
        return {"n": 42}

    return agent


class TestCheckpoint:
    def test_called_at_each_tool_step_with_consistent_snapshot(self) -> None:
        agent = _agent([_tool("compter", "c1"), _tool("compter", "c2"), _text("fini")])
        states: list[RunState] = []
        result = agent.run("vas-y", checkpoint=states.append)

        assert result.output == "fini"
        # one checkpoint per completed TOOL step; the final text-only step
        # produces the AgentResult, not a checkpoint
        assert [s.step for s in states] == [1, 2]
        # snapshot is consistent: the step's tool result is in the transcript
        assert states[0].messages[-1].role == "tool"
        assert states[0].messages[-1].tool_call_id == "c1"
        # snapshots are independent copies, not views of the live list
        assert len(states[0].messages) < len(states[1].messages)

    def test_called_after_post_turn_hook_correction(self) -> None:
        def hook(ctx):
            if ctx.correction_count == 0:
                return Message(role="user", content="précise !")
            return None

        agent = _agent([_text("court"), _text("plus long")], post_turn_hook=hook)
        states: list[RunState] = []
        result = agent.run("question", checkpoint=states.append)
        assert result.output == "plus long"
        assert [s.corrections for s in states] == [1]
        assert states[0].messages[-1].content == "précise !"

    def test_broken_checkpoint_callback_does_not_break_the_run(self) -> None:
        agent = _agent([_tool("compter", "c1"), _text("fini")])

        def boom(state):
            raise RuntimeError("disque plein")

        assert agent.run("vas-y", checkpoint=boom).output == "fini"


class TestRunStateSerialization:
    def test_json_roundtrip_is_lossless(self) -> None:
        agent = _agent([_tool("compter", "c1"), _text("fini")])
        states: list[RunState] = []
        agent.run("vas-y", checkpoint=states.append)

        restored = RunState.from_dict(json.loads(json.dumps(states[0].to_dict())))
        assert restored.step == states[0].step
        assert restored.turn_start == states[0].turn_start
        assert [m.to_dict() for m in restored.messages] == [
            m.to_dict() for m in states[0].messages
        ]
        # tool_calls survive (Message.to_dict 0.7.0 does the heavy lifting)
        assistant = [m for m in restored.messages if m.tool_calls]
        assert assistant and assistant[0].tool_calls[0].name == "compter"


class TestResume:
    def test_resume_from_checkpoint_finishes_the_run(self) -> None:
        first = _agent([_tool("compter", "c1"), _tool("compter", "c2"), _text("fini")])
        states: list[RunState] = []
        first.run("vas-y", checkpoint=states.append)

        # a NEW process restores the snapshot taken after step 1 and only
        # needs the remaining responses
        snapshot = RunState.from_dict(json.loads(json.dumps(states[0].to_dict())))
        second = _agent([_tool("compter", "c2"), _text("fini")])
        result = second.resume(snapshot)

        assert result.output == "fini"
        assert result.steps == 3  # step counting continues, it does not restart
        roles = [m.role for m in result.messages]
        assert roles.count("tool") == 2  # c1 (restored) + c2 (replayed)

    def test_max_steps_exception_carries_resumable_state(self) -> None:
        agent = _agent([_tool("compter", "c1"), _text("fini")], max_steps=1)
        with pytest.raises(MaxStepsExceeded) as exc_info:
            agent.run("vas-y")
        state = exc_info.value.state
        assert state.step == 1

        agent.max_steps = 3
        result = agent.resume(state)
        assert result.output == "fini"
        assert result.steps == 2

    def test_token_budget_exception_carries_resumable_state(self) -> None:
        usage = TokenUsage(input_tokens=60, output_tokens=40)  # 100/step
        agent = _agent(
            [_tool("compter", "c1", usage=usage), _text("fini", usage=usage)],
            token_budget=100,
        )
        with pytest.raises(TokenBudgetExceeded) as exc_info:
            agent.run("vas-y")
        state = exc_info.value.state
        assert state.input_tokens == 60 and state.output_tokens == 40

        agent.token_budget = 500
        result = agent.resume(state)
        assert result.output == "fini"
        assert result.usage.total_tokens == 200  # spend carried across the resume

    def test_cancel_carries_state_and_resume_continues(self) -> None:
        token = threading.Event()
        token.set()
        agent = _agent([_text("fini")])
        with pytest.raises(AgentCancelled) as exc_info:
            agent.run("vas-y", cancel_token=token)
        state = exc_info.value.state
        assert state.step == 0  # nothing had run yet

        result = agent.resume(state)
        assert result.output == "fini"

    def test_resume_skips_memory_compaction(self) -> None:
        class TruncatingMemory:
            def compact(self, messages):
                raise AssertionError("compact must not run on resume")

            def recall(self, query, k=5):
                return []

        agent = _agent([_text("fini")], memory=TruncatingMemory())
        state = RunState(
            messages=[
                Message(role="system", content="sys"),
                Message(role="user", content="question"),
            ],
            step=0,
            turn_start=2,
        )
        assert agent.resume(state).output == "fini"

    def test_resume_stream_emits_done(self) -> None:
        from autoagent.schema import StreamChunk

        class FakeStreamingProvider(FakeLLMProvider):
            def stream(self, request):
                yield StreamChunk(type="final", response=self.complete(request))

        agent = Agent(FakeStreamingProvider([_tool("compter", "c1"), _text("fini")]))

        @agent.tool
        def compter() -> dict:
            """Compte quelque chose."""
            return {"n": 42}

        state = RunState(
            messages=[
                Message(role="system", content="sys"),
                Message(role="user", content="question"),
            ],
            step=0,
            turn_start=2,
        )
        events = list(agent.resume_stream(state))
        done = [e for e in events if e.type == "done"]
        assert done and done[0].output == "fini"
