"""Record / replay déterministe des runs (0.16.0)."""

from __future__ import annotations

import pytest

from autoagent.agent import Agent
from autoagent.errors import ReplayMismatch
from autoagent.registry import ToolResult
from autoagent.replay import RecordSession, ReplaySession
from autoagent.schema import LLMResponse, ToolCall

from .conftest import FakeLLMProvider

SIDE_EFFECTS: list[str] = []


def _tool_call(name: str, call_id: str, **args) -> LLMResponse:
    return LLMResponse(content="", model="fake",
                       tool_calls=[ToolCall(id=call_id, name=name, arguments=args)])


def _text(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="fake")


@pytest.fixture(autouse=True)
def _clear():
    SIDE_EFFECTS.clear()


def _add_tools(agent: Agent) -> None:
    @agent.tool
    def incrementer(n: int) -> dict:
        """Ajoute 1 (avec effet de bord traçable)."""
        SIDE_EFFECTS.append(f"incrementer:{n}")
        return {"resultat": n + 1}


def _record(path, responses):
    """Enregistre un run complet ; renvoie l'AgentResult."""
    with RecordSession(path) as rec:
        agent = Agent(rec.provider(FakeLLMProvider(responses)), registry=rec.registry())
        _add_tools(agent)
        return agent.run("vas-y")


def test_record_writes_fixture(tmp_path) -> None:
    fx = tmp_path / "run.jsonl"
    _record(fx, [_tool_call("incrementer", "c1", n=41), _text("42")])
    lines = [l for l in fx.read_text(encoding="utf-8").splitlines() if l.strip()]
    kinds = [__import__("json").loads(l)["kind"] for l in lines]
    assert "llm" in kinds and "tool" in kinds
    assert SIDE_EFFECTS == ["incrementer:41"]     # l'outil a tourné en vrai à l'enregistrement


def test_full_replay_is_offline_and_deterministic(tmp_path) -> None:
    fx = tmp_path / "run.jsonl"
    original = _record(fx, [_tool_call("incrementer", "c1", n=41), _text("42")])
    SIDE_EFFECTS.clear()

    # provider de replay qui EXPLOSE s'il est réellement appelé (preuve zéro-réseau)
    class Exploding(FakeLLMProvider):
        def complete(self, request):
            raise AssertionError("le provider réel NE doit PAS être appelé en replay")

    with ReplaySession(fx) as rep:
        agent = Agent(rep.provider(), registry=rep.registry())
        # même si on ajoute le vrai outil, ReplayRegistry ne l'exécute pas
        @agent.tool
        def incrementer(n: int) -> dict:
            SIDE_EFFECTS.append("NE DEVRAIT PAS ARRIVER")
            return {"resultat": n + 1}
        result = agent.run("vas-y")

    assert result.output == original.output == "42"
    assert result.steps == original.steps
    assert SIDE_EFFECTS == []                       # ZÉRO effet de bord en replay total


def test_llm_only_replay_reexecutes_tools(tmp_path) -> None:
    fx = tmp_path / "run.jsonl"
    _record(fx, [_tool_call("incrementer", "c1", n=41), _text("42")])
    SIDE_EFFECTS.clear()

    with ReplaySession(fx) as rep:
        agent = Agent(rep.provider())               # PAS de registry rejoué → outils réels
        _add_tools(agent)
        result = agent.run("vas-y")

    assert result.output == "42"
    assert SIDE_EFFECTS == ["incrementer:41"]       # l'outil a été ré-exécuté


def test_serialization_roundtrip() -> None:
    resp = LLMResponse(content="ok", model="m",
                       tool_calls=[ToolCall(id="c1", name="f", arguments={"x": 1})],
                       reasoning_content="parce que")
    back = LLMResponse.from_dict(resp.to_dict())
    assert back.content == "ok" and back.model == "m"
    assert back.tool_calls[0].name == "f" and back.tool_calls[0].arguments == {"x": 1}
    assert back.reasoning_content == "parce que"

    res = ToolResult(ok=True, result={"a": [1, 2]}, error=None)
    assert ToolResult.from_dict(res.to_dict()) == res


def test_divergence_raises_mismatch(tmp_path) -> None:
    fx = tmp_path / "run.jsonl"
    _record(fx, [_tool_call("incrementer", "c1", n=41), _text("42")])

    # au replay, l'agent n'a PLUS l'outil « incrementer » → l'ensemble des tools
    # proposés diverge de l'enregistrement → ReplayMismatch (strict)
    with ReplaySession(fx) as rep:
        agent = Agent(rep.provider(), registry=rep.registry())  # aucun tool ajouté
        with pytest.raises(ReplayMismatch, match="divergence"):
            agent.run("vas-y")


def test_exhausted_fixture_raises_mismatch(tmp_path) -> None:
    from autoagent.replay import _Player
    from autoagent.schema import LLMRequest, Message
    fx = tmp_path / "one.jsonl"
    _record(fx, [_text("un seul tour")])              # fixture = 1 seul appel LLM
    player = _Player(fx, strict=False)                # positionnel : pas de check de signature
    req = LLMRequest(messages=[Message(role="user", content="x")])
    player.next_llm(req)                              # consomme l'unique événement
    with pytest.raises(ReplayMismatch, match="contient que|diverg"):
        player.next_llm(req)                          # épuisé → mismatch


def test_streaming_replay(tmp_path) -> None:
    from autoagent.schema import StreamChunk

    class FakeStreaming(FakeLLMProvider):
        def stream(self, request):
            yield StreamChunk(type="final", response=self.complete(request))

    fx = tmp_path / "stream.jsonl"
    with RecordSession(fx) as rec:
        agent = Agent(rec.provider(FakeStreaming([_text("bonjour")])), registry=rec.registry())
        list(agent.run_stream("salut"))

    with ReplaySession(fx) as rep:
        agent = Agent(rep.provider(), registry=rep.registry())
        done = [e for e in agent.run_stream("salut") if e.type == "done"]
    assert done and done[0].output == "bonjour"


def test_redaction_scrubs_secrets(tmp_path) -> None:
    fx = tmp_path / "secret.jsonl"
    with RecordSession(fx, redact=True) as rec:
        agent = Agent(rec.provider(FakeLLMProvider([_text("token: Bearer sk-ABC123DEF456GHI")])),
                      registry=rec.registry())
        agent.run("go")
    contenu = fx.read_text(encoding="utf-8")
    assert "sk-ABC123DEF456GHI" not in contenu       # le secret évident est scrubé
