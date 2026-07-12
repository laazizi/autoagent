"""SummarizingMemory + token_budget (0.10.0).

La mémoire résumante replie les vieux tours dans un résumé LLM
incrémental au lieu de les jeter ; le budget tokens borne un run sur
l'usage RAPPORTÉ par le provider.
"""

from __future__ import annotations

import pytest

from autoagent import Agent, SummarizingMemory, TokenBudgetExceeded
from autoagent.schema import LLMRequest, LLMResponse, Message, ModelConfig, TokenUsage, ToolCall


class _ScriptedProvider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.config = ModelConfig(provider="fake", model="fake-model")
        self._responses = list(responses)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self._responses.pop(0)


def _convo(n_pairs: int) -> list[Message]:
    messages = [Message(role="system", content="Tu es un assistant.")]
    for i in range(n_pairs):
        messages.append(Message(role="user", content=f"question numéro {i} sur le capteur CMP-{i}"))
        messages.append(Message(role="assistant", content=f"réponse {i}"))
    return messages


class TestSummarizingMemory:
    def test_below_threshold_unchanged(self) -> None:
        provider = _ScriptedProvider([])
        memory = SummarizingMemory(provider, max_messages=40, keep_recent=12)
        messages = _convo(5)
        assert memory.compact(messages) == messages
        assert provider.requests == []  # aucun appel LLM sous le seuil

    def test_folds_old_turns_into_summary(self) -> None:
        provider = _ScriptedProvider([LLMResponse(content="RÉSUMÉ: capteurs 0..13 traités")])
        memory = SummarizingMemory(provider, max_messages=20, keep_recent=6)
        compacted = memory.compact(_convo(15))  # 30 messages non-système

        assert len(provider.requests) == 1  # un seul appel de résumé
        summary_msgs = [m for m in compacted if m.content.startswith("[Résumé")]
        assert len(summary_msgs) == 1 and "capteurs 0..13" in summary_msgs[0].content
        non_system = [m for m in compacted if m.role != "system"]
        assert non_system[0].role == "user"  # coupe alignée sur un user
        assert len(non_system) <= 8  # keep_recent (~6, aligné)
        # recall lexical retrouve un détail replié
        recalled = memory.recall("question capteur CMP-3")
        assert any("CMP-3" in m.content for m in recalled)

    def test_incremental_second_compaction(self) -> None:
        provider = _ScriptedProvider(
            [LLMResponse(content="RÉSUMÉ v1"), LLMResponse(content="RÉSUMÉ v2")]
        )
        memory = SummarizingMemory(provider, max_messages=20, keep_recent=6)
        history = _convo(15)
        memory.compact(history)
        history += _convo(4)[1:]  # +8 messages (sans dupliquer le system)
        memory.compact(history)
        assert len(provider.requests) == 2
        # le 2e appel ne re-résume pas tout : il porte le résumé précédent
        assert "RÉSUMÉ v1" in provider.requests[1].messages[-1].content

    def test_provider_failure_skips_compaction(self) -> None:
        class _Broken:
            config = ModelConfig(provider="fake", model="fake")

            def complete(self, request: LLMRequest) -> LLMResponse:
                raise RuntimeError("LLM down")

        memory = SummarizingMemory(_Broken(), max_messages=20, keep_recent=6)
        messages = _convo(15)
        assert memory.compact(messages) == messages  # inchangé, jamais tronqué en silence

    def test_inband_summary_reabsorbed_not_duplicated(self) -> None:
        # L'hôte persiste l'historique compacté -> notre résumé revient en
        # message système. Une compaction suivante ne doit PAS l'empiler.
        provider = _ScriptedProvider([LLMResponse(content="RÉSUMÉ v2")])
        memory = SummarizingMemory(provider, max_messages=20, keep_recent=6)
        persisted = [
            Message(role="system", content="Tu es un assistant."),
            Message(role="system", content="[Résumé de la conversation antérieure]\nRÉSUMÉ v1"),
            *_convo(15)[1:],
        ]
        compacted = memory.compact(persisted)
        summaries = [m for m in compacted if m.content.startswith("[Résumé")]
        assert len(summaries) == 1
        assert "RÉSUMÉ v1" in provider.requests[0].messages[-1].content  # graine réabsorbée


class TestTokenBudget:
    def _usage_response(self, content: str, tool: bool = False) -> LLMResponse:
        return LLMResponse(
            content=content,
            tool_calls=[ToolCall(id="c1", name="noop", arguments={})] if tool else [],
            usage=TokenUsage(input_tokens=600, output_tokens=100),
        )

    def test_budget_stops_before_next_call(self) -> None:
        provider = _ScriptedProvider(
            [self._usage_response("", tool=True), self._usage_response("done")]
        )
        agent = Agent(provider, token_budget=500)

        @agent.tool
        def noop() -> dict:
            """No-op."""
            return {}

        with pytest.raises(TokenBudgetExceeded) as exc_info:
            agent.run("go")
        assert getattr(exc_info.value, "spent", 0) == 700  # 600+100 du 1er appel
        assert len(provider.requests) == 1  # le 2e appel n'a jamais été émis

    def test_result_carries_aggregated_usage(self) -> None:
        provider = _ScriptedProvider(
            [self._usage_response("", tool=True), self._usage_response("done")]
        )
        agent = Agent(provider)  # pas de budget

        @agent.tool
        def noop() -> dict:
            """No-op."""
            return {}

        result = agent.run("go")
        assert result.usage is not None
        assert result.usage.input_tokens == 1200  # 2 appels x 600
        assert result.usage.output_tokens == 200
        assert result.usage.total_tokens == 1400

    def test_stream_yields_budget_error_event(self) -> None:
        provider = _ScriptedProvider([self._usage_response("", tool=True)])
        agent = Agent(provider, token_budget=500)

        @agent.tool
        def noop() -> dict:
            """No-op."""
            return {}

        provider.stream = lambda request: iter(  # type: ignore[attr-defined]
            [__import__("autoagent.schema", fromlist=["StreamChunk"]).StreamChunk(
                type="final", response=provider.complete(request))]
        )
        events = list(agent.run_stream("go"))
        assert events[-1].type == "error"
        assert "token_budget=500" in events[-1].error
