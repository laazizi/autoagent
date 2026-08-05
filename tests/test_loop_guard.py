"""`max_repeated_tool_calls` — garde anti-boucle par le CODE (0.18.0).

Trou avéré avant ce correctif : un agent qui redemande 40 fois le même appel avec
les mêmes arguments consommait `max_steps` ET tout le `token_budget` au tarif
plein, ré-exécutait l'effet de bord à chaque tour, et finissait sur un `max_steps`
muet — non diagnosticable dans la trace.

Le remède n'est pas une supplication de prompt : la répétition est détectée par le
code et le modèle reçoit un refus déterministe sur le MÊME canal qu'un refus de
politique — le chemin de replanification qui fonctionne déjà.
"""

from __future__ import annotations

from .conftest import FakeLLMProvider

from autoagent import Agent, TraceEmitter
from autoagent.agent import _call_signature, _count_call_signatures
from autoagent.schema import LLMResponse, Message, ToolCall


def _repete(nom: str, args: dict, n: int) -> list:
    """n réponses qui redemandent TOUJOURS le même appel, puis un texte final."""
    return [
        LLMResponse(tool_calls=[ToolCall(id=f"c{i}", name=nom, arguments=dict(args))])
        for i in range(n)
    ] + ["j'arrête"]


class TestSignature:
    def test_ordre_des_cles_indifferent(self) -> None:
        a = ToolCall(id="1", name="t", arguments={"x": 1, "y": 2})
        b = ToolCall(id="2", name="t", arguments={"y": 2, "x": 1})
        assert _call_signature(a) == _call_signature(b)

    def test_arguments_differents_signatures_differentes(self) -> None:
        a = ToolCall(id="1", name="t", arguments={"x": 1})
        b = ToolCall(id="2", name="t", arguments={"x": 2})
        assert _call_signature(a) != _call_signature(b)

    def test_comptage_depuis_le_transcript(self) -> None:
        msgs = [
            Message(role="assistant", content="",
                    tool_calls=[ToolCall(id="a", name="t", arguments={"x": 1})]),
            Message(role="assistant", content="",
                    tool_calls=[ToolCall(id="b", name="t", arguments={"x": 1})]),
            Message(role="assistant", content="",
                    tool_calls=[ToolCall(id="c", name="t", arguments={"x": 9})]),
        ]
        counts = _count_call_signatures(msgs)
        assert sorted(counts.values()) == [1, 2]


class TestGarde:
    def test_desactivee_par_defaut(self) -> None:
        """Rétrocompatibilité : sans l'option, l'outil est ré-exécuté à l'infini."""
        appels = {"n": 0}
        agent = Agent(FakeLLMProvider(_repete("boucle", {}, 5)), max_steps=8)

        @agent.tool
        def boucle() -> dict:
            """Outil qui ne fait pas avancer."""
            appels["n"] += 1
            return {"encore": True}

        agent.run("vas-y")
        assert appels["n"] == 5           # tout a été exécuté, aucune garde

    def test_l_effet_de_bord_cesse_apres_le_seuil(self) -> None:
        """LE gain majeur : on arrête de RÉ-EXÉCUTER (mail, POST, sous-agent…)."""
        appels = {"n": 0}
        agent = Agent(FakeLLMProvider(_repete("boucle", {"x": 1}, 6)),
                      max_steps=8, max_repeated_tool_calls=2)

        @agent.tool
        def boucle(x: int) -> dict:
            """Outil qui ne fait pas avancer."""
            appels["n"] += 1
            return {"encore": True}

        res = agent.run("vas-y")
        assert appels["n"] == 2, f"exécuté {appels['n']} fois au lieu de 2"
        assert any("RepeatedCall" in (m.content or "") for m in res.messages)

    def test_le_modele_recoit_un_refus_actionnable(self) -> None:
        agent = Agent(FakeLLMProvider(_repete("boucle", {}, 4)),
                      max_steps=8, max_repeated_tool_calls=1)

        @agent.tool
        def boucle() -> dict:
            """Outil qui ne fait pas avancer."""
            return {"encore": True}

        res = agent.run("vas-y")
        refus = next(m for m in res.messages if "RepeatedCall" in (m.content or ""))
        assert "boucle" in refus.content
        # le message dit QUOI FAIRE, pas seulement « non »
        assert "Change the arguments" in refus.content

    def test_arguments_differents_ne_sont_pas_bloques(self) -> None:
        """Un agent qui PROGRESSE (arguments qui varient) n'est jamais gêné."""
        appels = {"n": 0}
        agent = Agent(FakeLLMProvider([
            LLMResponse(tool_calls=[ToolCall(id=f"c{i}", name="page",
                                             arguments={"p": i})]) for i in range(5)
        ] + ["fini"]), max_steps=8, max_repeated_tool_calls=1)

        @agent.tool
        def page(p: int) -> dict:
            """Pagination : chaque appel est différent."""
            appels["n"] += 1
            return {"page": p}

        agent.run("parcours les pages")
        assert appels["n"] == 5

    def test_trace_nomme_l_echec(self) -> None:
        """La trace doit DIRE « boucle bloquée » au lieu d'un max_steps muet."""
        vus: list[str] = []
        trace = TraceEmitter(on_event=lambda ev: vus.append(ev.type))
        agent = Agent(FakeLLMProvider(_repete("boucle", {}, 4)),
                      max_steps=8, max_repeated_tool_calls=1, trace=trace)

        @agent.tool
        def boucle() -> dict:
            """Outil qui ne fait pas avancer."""
            return {"encore": True}

        agent.run("vas-y")
        assert "loop_guard_block" in vus

    def test_compatible_appels_paralleles(self) -> None:
        appels = {"n": 0}
        # Deux appels IDENTIQUES dans le MÊME tour : le 1er passe, le 2e est du bruit.
        agent = Agent(FakeLLMProvider([
            LLMResponse(tool_calls=[
                ToolCall(id="a", name="boucle", arguments={"x": 1}),
                ToolCall(id="b", name="boucle", arguments={"x": 1}),
            ]),
            LLMResponse(tool_calls=[ToolCall(id="c", name="boucle", arguments={"x": 1})]),
            "fini",
        ]), max_steps=6, max_repeated_tool_calls=2, parallel_tool_calls=True)

        @agent.tool
        def boucle(x: int) -> dict:
            """Outil répété."""
            appels["n"] += 1
            return {"ok": True}

        res = agent.run("vas-y")
        # Les 2 du premier tour comptent comme 2 demandes → le 3e est bloqué.
        assert appels["n"] == 2
        assert any("RepeatedCall" in (m.content or "") for m in res.messages)
