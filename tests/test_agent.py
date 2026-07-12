from typing import Any

import pytest

from autoagent.agent import Agent
from autoagent.errors import MaxStepsExceeded
from autoagent.schema import LLMResponse, Message, ToolCall, ToolSpec

from .conftest import FakeLLMProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_response(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="fake")


def _tool_response(tool_calls: list[ToolCall], content: str = "") -> LLMResponse:
    return LLMResponse(content=content, tool_calls=tool_calls, model="fake")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentBasicRun:
    def test_simple_text_response(self) -> None:
        provider = FakeLLMProvider([_text_response("Hello, world!")])
        agent = Agent(provider)
        result = agent.run("Say hi")
        assert result.output == "Hello, world!"
        assert result.steps == 1

    def test_run_messages(self) -> None:
        provider = FakeLLMProvider([_text_response("OK")])
        agent = Agent(provider)
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="What is 2+2?"),
        ]
        result = agent.run_messages(messages)
        assert result.output == "OK"

    def test_messages_preserved(self) -> None:
        provider = FakeLLMProvider([_text_response("done")])
        agent = Agent(provider)
        result = agent.run("prompt")
        assert len(result.messages) >= 3  # system, user, assistant
        roles = [m.role for m in result.messages]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles


class TestAgentWithTools:
    def test_single_tool_call(self) -> None:
        provider = FakeLLMProvider(
            [
                _tool_response([ToolCall(id="c1", name="add", arguments={"a": 3, "b": 4})]),
                _text_response("The sum is 7."),
            ]
        )

        agent = Agent(provider)

        def add(a: int, b: int) -> int:
            return a + b

        agent.registry.add(ToolSpec(name="add", description="Add"), add)

        result = agent.run("What is 3+4?")
        assert result.output == "The sum is 7."
        assert result.steps == 2

    def test_tool_result_in_messages(self) -> None:
        provider = FakeLLMProvider(
            [
                _tool_response([ToolCall(id="c1", name="get_name", arguments={})]),
                _text_response("Your name is Claude."),
            ]
        )

        agent = Agent(provider)

        def get_name(context: dict[str, Any] | None = None) -> str:
            return "Claude"

        agent.registry.add(ToolSpec(name="get_name", description="Get name"), get_name)

        result = agent.run("Who am I?")
        tool_messages = [m for m in result.messages if m.role == "tool"]
        assert len(tool_messages) == 1
        assert '"Claude"' in tool_messages[0].content or "Claude" in tool_messages[0].content


class TestAgentMaxSteps:
    def test_exceeds_max_steps(self) -> None:
        # Always return tool calls -> never a text response -> max_steps exceeded
        responses = [_tool_response([ToolCall(id=f"c{i}", name="noop", arguments={})]) for i in range(10)]
        provider = FakeLLMProvider(responses)

        agent = Agent(provider, max_steps=3)

        def noop(context: dict[str, Any] | None = None) -> str:
            return "done"

        agent.registry.add(ToolSpec(name="noop", description="No-op"), noop)

        with pytest.raises(MaxStepsExceeded):
            agent.run("loop forever")


class TestAgentDynamicTools:
    def _make_fake_builder(self) -> Any:
        """Build a DynamicToolBuilder that actually registers a working tool
        each time `build` is called, so we can exercise the real budget
        mechanism end-to-end instead of just checking registration."""
        from autoagent.dynamic import DynamicToolBuilder
        from autoagent.schema import ToolSpec as TS

        class _FakeGeneratedTool:
            def __init__(self, name: str) -> None:
                self.spec = TS(
                    name=name,
                    description=f"fake tool {name}",
                    input_schema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                )

            def __call__(self, **kwargs: Any) -> dict[str, Any]:
                return {"called": self.spec.name}

        class FakeBuilder(DynamicToolBuilder):
            def __init__(self) -> None:
                self.build_count = 0

            def build(self, request: Any) -> Any:
                self.build_count += 1
                return _FakeGeneratedTool(name=f"generated_{self.build_count}")

        return FakeBuilder()

    def test_create_python_tool_registered_after_enable(self) -> None:
        provider = FakeLLMProvider([_text_response("ok")])
        agent = Agent(provider, max_dynamic_tools_per_run=3)
        agent.enable_dynamic_tools(self._make_fake_builder())
        assert "create_python_tool" in agent.registry

    def test_dynamic_tool_budget_blocks_after_max(self) -> None:
        """First N calls succeed, call N+1 must fail with a budget error."""
        # Sequence: agent calls create_python_tool 4 times, then emits text.
        responses = [
            _tool_response(
                [ToolCall(id=f"c{i}", name="create_python_tool", arguments={"capability": f"cap-{i}"})]
            )
            for i in range(1, 5)
        ]
        responses.append(_text_response("done"))

        provider = FakeLLMProvider(responses)
        agent = Agent(provider, max_dynamic_tools_per_run=3, max_steps=10)
        builder = self._make_fake_builder()
        agent.enable_dynamic_tools(builder)

        result = agent.run("create some tools")

        # Builder.build was called at most max_dynamic_tools_per_run times.
        assert builder.build_count == 3, f"Expected builder.build called 3 times, got {builder.build_count}"
        # The 4th create_python_tool invocation must surface a budget error
        # in the tool messages, not crash the agent.
        tool_messages = [m for m in result.messages if m.role == "tool"]
        last_tool_msg = tool_messages[-1]
        assert "budget" in last_tool_msg.content.lower()


class TestAgentFromModel:
    def test_from_model(self) -> None:
        agent = Agent.from_model("openai", "gpt-4")
        assert agent.provider.config.provider == "openai"
        assert agent.provider.config.model == "gpt-4"

    def test_from_model_config(self) -> None:
        from autoagent.schema import ModelConfig

        config = ModelConfig(provider="anthropic", model="claude-3")
        agent = Agent.from_model_config(config)
        assert agent.provider.config.provider == "anthropic"
        assert agent.provider.config.model == "claude-3"


class TestAgentSystemPrompt:
    def test_default_system_prompt_set(self) -> None:
        provider = FakeLLMProvider([_text_response("ok")])
        agent = Agent(provider)
        agent.run("test")
        # Provider should have received messages with the default system prompt
        assert len(provider.calls) == 1
        sent_messages = provider.calls[0].messages
        assert sent_messages[0].role == "system"
        assert "AI agent" in sent_messages[0].content

    def test_custom_system_prompt(self) -> None:
        provider = FakeLLMProvider([_text_response("ok")])
        agent = Agent(provider, system_prompt="Custom prompt.")
        agent.run("test")
        sent_messages = provider.calls[0].messages
        assert sent_messages[0].content == "Custom prompt."
