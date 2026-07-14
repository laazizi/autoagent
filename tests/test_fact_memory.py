"""FactMemory — extraction/consolidation de faits + register_remember_tool (0.12.0)."""

from __future__ import annotations

import json

import pytest

from autoagent.agent import Agent
from autoagent.memory import FactMemory
from autoagent.schema import LLMResponse, Message, ToolCall

from .conftest import FakeLLMProvider


def _ops(*operations) -> LLMResponse:
    return LLMResponse(content=json.dumps({"operations": list(operations)}), model="fake")


def _convo(n_pairs: int) -> list[Message]:
    messages = [Message(role="system", content="sys")]
    for i in range(n_pairs):
        messages.append(Message(role="user", content=f"question numéro {i}"))
        messages.append(Message(role="assistant", content=f"réponse numéro {i}"))
    return messages


class TestCompactExtraction:
    def test_folds_old_turns_into_facts_message(self) -> None:
        provider = FakeLLMProvider([
            _ops({"op": "add", "fact": "préfère être rappelé le soir", "subject": "rdv"}),
        ])
        memory = FactMemory(provider, max_messages=6, keep_recent=2)
        result = memory.compact(_convo(8))  # 16 messages non-système > 6

        facts_msgs = [m for m in result if (m.content or "").startswith("[Faits mémorisés]")]
        assert len(facts_msgs) == 1
        assert "préfère être rappelé le soir (rdv)" in facts_msgs[0].content
        # la fenêtre récente reste verbatim et commence par un user
        tail = [m for m in result if m.role != "system"]
        assert tail[0].role == "user"
        assert memory.facts()[0]["fact"] == "préfère être rappelé le soir"

    def test_update_replaces_contradicted_fact(self) -> None:
        provider = FakeLLMProvider([
            _ops({"op": "add", "fact": "préfère être rappelé le soir"}),
            _ops({"op": "update", "id": 1, "fact": "préfère être rappelé le matin"}),
        ])
        memory = FactMemory(provider, max_messages=4, keep_recent=2)
        memory.compact(_convo(4))
        memory.compact(_convo(8))

        facts = memory.facts()
        assert len(facts) == 1  # PAS d'empilement
        assert facts[0]["fact"] == "préfère être rappelé le matin"

    def test_delete_removes_fact(self) -> None:
        provider = FakeLLMProvider([
            _ops({"op": "add", "fact": "a un abonnement premium"}),
            _ops({"op": "delete", "id": 1}),
        ])
        memory = FactMemory(provider, max_messages=4, keep_recent=2)
        memory.compact(_convo(4))
        memory.compact(_convo(8))
        assert memory.facts() == []

    def test_extraction_failure_skips_compaction(self) -> None:
        class BrokenProvider(FakeLLMProvider):
            def complete(self, request):
                raise RuntimeError("réseau")

        memory = FactMemory(BrokenProvider(), max_messages=4, keep_recent=2)
        original = _convo(6)
        assert memory.compact(original) == original  # inchangé, pas d'exception

    def test_malformed_operations_are_ignored(self) -> None:
        provider = FakeLLMProvider([
            LLMResponse(content="pas du json", model="fake"),
            _ops({"op": "update", "id": 999, "fact": "x"}, {"op": "???"}, "pas-un-dict"),
        ])
        memory = FactMemory(provider, max_messages=4, keep_recent=2)
        memory.compact(_convo(4))
        memory.compact(_convo(8))
        assert memory.facts() == []  # rien appliqué, rien cassé

    def test_json_fences_tolerated(self) -> None:
        fenced = "```json\n" + json.dumps(
            {"operations": [{"op": "add", "fact": "habite Lyon"}]}
        ) + "\n```"
        provider = FakeLLMProvider([LLMResponse(content=fenced, model="fake")])
        memory = FactMemory(provider, max_messages=4, keep_recent=2)
        memory.compact(_convo(4))
        assert memory.facts()[0]["fact"] == "habite Lyon"

    def test_new_conversation_of_similar_length_is_extracted(self) -> None:
        """Bug trouvé en test réel : un 2e APPEL (nouvelle conversation, longueur
        similaire) était silencieusement ignoré parce que `_covered` croyait le
        préfixe déjà traité — les contradictions du rappel étaient perdues."""
        provider = FakeLLMProvider([
            _ops({"op": "add", "fact": "préfère le soir"}),
            _ops({"op": "update", "id": 1, "fact": "préfère le matin"}),
        ])
        memory = FactMemory(provider, max_messages=6, keep_recent=2)
        memory.compact(_convo(4))          # appel 1 (8 messages non-système)

        appel2 = [Message(role="system", content="sys")]
        for i in range(4):                 # NOUVELLE conversation, même longueur
            appel2.append(Message(role="user", content=f"rappel, échange {i}"))
            appel2.append(Message(role="assistant", content=f"ok {i}"))
        memory.compact(appel2)             # appel 2 : DOIT extraire

        facts = memory.facts()
        assert len(facts) == 1 and facts[0]["fact"] == "préfère le matin"

    def test_inband_facts_message_not_duplicated(self) -> None:
        provider = FakeLLMProvider([_ops({"op": "add", "fact": "2 véhicules au foyer"})])
        memory = FactMemory(provider, max_messages=4, keep_recent=2)
        compacted = memory.compact(_convo(4))
        # l'hôte persiste puis repasse l'historique compacté
        again = memory.compact(list(compacted))
        markers = [m for m in again if (m.content or "").startswith("[Faits mémorisés]")]
        assert len(markers) == 1


class TestRememberRecall:
    def test_remember_direct_and_dedupe(self) -> None:
        memory = FactMemory(FakeLLMProvider())
        first = memory.remember("part en vacances en août", subject="agenda")
        second = memory.remember("Part en vacances en AOÛT")  # même fait
        assert first["id"] == second["id"]
        assert len(memory.facts()) == 1

    def test_recall_is_lexical_over_facts(self) -> None:
        memory = FactMemory(FakeLLMProvider())
        memory.remember("préfère être rappelé le matin", subject="rdv")
        memory.remember("numéro fixe 0472000000", subject="contact")
        matches = memory.recall("quel numéro de téléphone ?")
        assert len(matches) == 1 and "0472000000" in matches[0].content

    def test_forget(self) -> None:
        memory = FactMemory(FakeLLMProvider())
        stored = memory.remember("fait temporaire")
        assert memory.forget(stored["id"]) is True
        assert memory.forget(999) is False
        assert memory.facts() == []


class TestPersistence:
    def test_json_roundtrip_across_instances(self, tmp_path) -> None:
        store = tmp_path / "caller_0601020304.json"
        memory = FactMemory(FakeLLMProvider(), path=store)
        memory.remember("préfère le matin", subject="rdv")

        reloaded = FactMemory(FakeLLMProvider(), path=store)
        assert reloaded.facts()[0]["fact"] == "préfère le matin"
        # les ids continuent, pas de collision
        new = reloaded.remember("autre fait")
        assert new["id"] == 2

    def test_corrupt_store_starts_empty(self, tmp_path) -> None:
        store = tmp_path / "bad.json"
        store.write_text("{pas du json", encoding="utf-8")
        memory = FactMemory(FakeLLMProvider(), path=store)
        assert memory.facts() == []


class TestRememberTool:
    def test_agent_stores_fact_through_the_tool(self) -> None:
        memory = FactMemory(FakeLLMProvider())
        provider = FakeLLMProvider([
            LLMResponse(content="", model="fake", tool_calls=[
                ToolCall(id="c1", name="remember",
                         arguments={"fact": "part en vacances en août", "subject": "agenda"}),
            ]),
            LLMResponse(content="noté !", model="fake"),
        ])
        agent = Agent(provider, memory=memory)
        agent.register_remember_tool()
        result = agent.run("note que je pars en vacances en août")

        assert result.output == "noté !"
        assert memory.facts()[0]["fact"] == "part en vacances en août"
        tool_msg = [m for m in result.messages if m.role == "tool"][0]
        assert '"stored": true' in tool_msg.content

    def test_noop_without_fact_capable_memory(self) -> None:
        agent = Agent(FakeLLMProvider([]))
        agent.register_remember_tool()          # memory=None → silencieux
        assert "remember" not in agent.registry

        from autoagent.memory import BufferMemory
        agent2 = Agent(FakeLLMProvider([]), memory=BufferMemory())
        agent2.register_remember_tool()         # pas de .remember → silencieux
        assert "remember" not in agent2.registry
