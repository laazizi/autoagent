"""tool_policy — allow / deny / ask-user (approval gate) / fail-closed (0.11.0)."""

from __future__ import annotations

import json

import pytest

from autoagent.agent import Agent, RunState, ToolPolicyContext
from autoagent.errors import ApprovalRequired
from autoagent.schema import LLMResponse, Message, ToolCall

from .conftest import FakeLLMProvider


def _text(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="fake")


def _tool(name: str, call_id: str, **arguments) -> LLMResponse:
    return LLMResponse(
        content="", model="fake",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
    )


EXECUTED: list[str] = []


def _agent(responses, policy, **kwargs) -> Agent:
    agent = Agent(FakeLLMProvider(responses), tool_policy=policy, **kwargs)

    @agent.tool(permissions=["filesystem.write"])
    def effacer(chemin: str) -> dict:
        """Efface un fichier (sensible)."""
        EXECUTED.append(f"effacer:{chemin}")
        return {"efface": chemin}

    @agent.tool
    def lire(chemin: str) -> dict:
        """Lit un fichier (anodin)."""
        EXECUTED.append(f"lire:{chemin}")
        return {"contenu": "..."}

    return agent


@pytest.fixture(autouse=True)
def _clear_executed():
    EXECUTED.clear()


class TestAllowDeny:
    def test_none_allows(self) -> None:
        agent = _agent([_tool("lire", "c1", chemin="a.txt"), _text("fini")], policy=lambda ctx: None)
        assert agent.run("vas-y").output == "fini"
        assert EXECUTED == ["lire:a.txt"]

    def test_str_denies_and_model_sees_the_reason(self) -> None:
        def policy(ctx: ToolPolicyContext):
            if "filesystem.write" in (ctx.spec.permissions if ctx.spec else []):
                return "écriture interdite pour cet utilisateur"
            return None

        provider_responses = [_tool("effacer", "c1", chemin="x"), _text("compris")]
        agent = _agent(provider_responses, policy)
        result = agent.run("efface x")

        assert EXECUTED == []  # jamais exécuté
        tool_msgs = [m for m in result.messages if m.role == "tool"]
        assert "ToolPolicyDenied" in tool_msgs[0].content
        assert "écriture interdite" in tool_msgs[0].content

    def test_policy_context_carries_call_spec_step(self) -> None:
        seen = {}

        def policy(ctx: ToolPolicyContext):
            seen.update(name=ctx.call.name, args=ctx.call.arguments,
                        perms=ctx.spec.permissions, step=ctx.step,
                        user=ctx.context.get("user"))
            return None

        agent = _agent([_tool("effacer", "c1", chemin="x"), _text("ok")], policy)
        agent.run("go", context={"user": "mo"})
        assert seen == {"name": "effacer", "args": {"chemin": "x"},
                        "perms": ["filesystem.write"], "step": 1, "user": "mo"}

    def test_buggy_policy_fails_closed(self) -> None:
        def policy(ctx):
            raise RuntimeError("base d'autorisations injoignable")

        agent = _agent([_tool("lire", "c1", chemin="a"), _text("ok")], policy)
        result = agent.run("go")
        assert EXECUTED == []  # refusé, pas autorisé par défaut
        tool_msgs = [m for m in result.messages if m.role == "tool"]
        assert "ToolPolicyDenied" in tool_msgs[0].content

    def test_mixed_allow_deny_in_parallel_turn(self) -> None:
        both = LLMResponse(content="", model="fake", tool_calls=[
            ToolCall(id="c1", name="lire", arguments={"chemin": "a"}),
            ToolCall(id="c2", name="effacer", arguments={"chemin": "b"}),
        ])

        def policy(ctx):
            return "non" if ctx.call.name == "effacer" else None

        agent = _agent([both, _text("ok")], policy, parallel_tool_calls=True)
        result = agent.run("go")
        assert EXECUTED == ["lire:a"]
        tool_msgs = {m.tool_call_id: m.content for m in result.messages if m.role == "tool"}
        assert "ToolPolicyDenied" in tool_msgs["c2"] and "contenu" in tool_msgs["c1"]


class TestApprovalGate:
    def _asking_policy(self, approved: set):
        def policy(ctx: ToolPolicyContext):
            if ctx.call.name != "effacer":
                return None
            if ctx.call.id in approved:
                return None
            raise ApprovalRequired(f"effacement de {ctx.call.arguments.get('chemin')}")
        return policy

    def test_pause_before_any_side_effect_then_resume_after_approval(self) -> None:
        approved: set = set()
        policy = self._asking_policy(approved)
        agent = _agent([_tool("effacer", "c1", chemin="x"), _text("fini")], policy)

        with pytest.raises(ApprovalRequired) as exc_info:
            agent.run("efface x")
        exc = exc_info.value
        assert EXECUTED == []                       # pause AVANT tout effet de bord
        assert [c.name for c in exc.calls] == ["effacer"]
        assert exc.state.messages[-1].role == "assistant"  # LLM call dans le transcript

        # le snapshot survit à un aller-retour JSON (autre process)
        state = RunState.from_dict(json.loads(json.dumps(exc.state.to_dict())))

        approved.add("c1")                          # l'humain valide
        result = agent.resume(state)
        assert EXECUTED == ["effacer:x"]            # exécuté UNE fois, à la reprise
        assert result.output == "fini"

    def test_unapproved_resume_pauses_again(self) -> None:
        policy = self._asking_policy(approved=set())
        agent = _agent([_tool("effacer", "c1", chemin="x")], policy)
        with pytest.raises(ApprovalRequired) as first:
            agent.run("efface x")
        with pytest.raises(ApprovalRequired):
            agent.resume(first.value.state)         # idempotent
        assert EXECUTED == []

    def test_rejected_resume_surfaces_denial_to_model(self) -> None:
        def policy(ctx):
            if ctx.call.id == "c1" and ctx.context.get("rejected"):
                return "refusé par l'opérateur"
            if ctx.call.name == "effacer":
                raise ApprovalRequired("validation requise")
            return None

        agent = _agent([_tool("effacer", "c1", chemin="x"), _text("compris")], policy)
        with pytest.raises(ApprovalRequired) as exc_info:
            agent.run("efface x")

        result = agent.resume(exc_info.value.state, context={"rejected": True})
        assert EXECUTED == []
        tool_msgs = [m for m in result.messages if m.role == "tool"]
        assert "refusé par l'opérateur" in tool_msgs[0].content
        assert result.output == "compris"

    def test_streaming_yields_terminal_event_with_state(self) -> None:
        from autoagent.schema import StreamChunk

        class FakeStreamingProvider(FakeLLMProvider):
            def stream(self, request):
                yield StreamChunk(type="final", response=self.complete(request))

        policy = self._asking_policy(approved=set())
        agent = Agent(FakeStreamingProvider([_tool("effacer", "c1", chemin="x")]),
                      tool_policy=policy)

        @agent.tool
        def effacer(chemin: str) -> dict:
            """Efface un fichier (sensible)."""
            EXECUTED.append(f"effacer:{chemin}")
            return {"efface": chemin}

        events = list(agent.run_stream("efface x"))
        last = events[-1]
        assert last.type == "error" and last.error.startswith("approval_required")
        assert last.state is not None and last.state.messages
