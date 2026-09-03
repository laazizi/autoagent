"""Exécution AU FIL DU FLUX des outils idempotents (0.21.0).

Un chunk `tool_call` arrive dès qu'un appel est complet — souvent bien avant la
fin du message. La boucle lance l'outil tout de suite si, et seulement si, il
est déclaré `idempotent=True` et que les gardes le laissent passer. PASTE
(arXiv 2603.18897) mesure -43 % de temps par tâche avec ce recouvrement ; le
papier compagnon (2606.07846) fixe la règle : JAMAIS sur un outil à effet de
bord.

Ces tests verrouillent, dans l'ordre :

  1. LE RECOUVREMENT EST RÉEL — l'outil a DÉMARRÉ avant que le fournisseur ait
     émis `final` (horodatages, pas supposition).
  2. UN SEUL APPEL — le chemin normal CONSOMME le résultat, il ne relance pas.
  3. SANS `idempotent`, RIEN NE PART EN AVANCE — un effet de bord attend.
  4. LES GARDES PASSENT AVANT — un outil refusé par la politique n'est pas
     lancé en avance, et le refus apparaît comme d'habitude.
  5. Transcript et événements IDENTIQUES à la voie sans anticipation.
  6. Un résultat anticipé orphelin (absent du `final`) est jeté sans dégât.
"""

from __future__ import annotations

import threading
import time

from autoagent import Agent, TraceEmitter
from autoagent.providers.base import LLMProvider
from autoagent.schema import LLMResponse, ModelConfig, StreamChunk, TokenUsage, ToolCall


class Flux(LLMProvider):
    """Émet ses `tool_call` puis ATTEND `delai` avant `final` — comme un modèle
    qui continue de parler après avoir décidé ses appels."""

    def __init__(self, tours: list[list[ToolCall] | str], *, delai: float = 0.25,
                 orphelin: ToolCall | None = None) -> None:
        super().__init__(ModelConfig(provider="f", model="f", api_key="x"))
        self.tours = list(tours)
        self.delai = delai
        self.orphelin = orphelin
        self.final_emis_a: list[float] = []

    def complete(self, request):  # type: ignore[no-untyped-def]
        tour = self.tours.pop(0)
        if isinstance(tour, str):
            return LLMResponse(content=tour, usage=TokenUsage(input_tokens=10, output_tokens=2))
        return LLMResponse(tool_calls=list(tour), usage=TokenUsage(input_tokens=10, output_tokens=2))

    def stream(self, request):  # type: ignore[no-untyped-def]
        tour = self.tours.pop(0)
        if isinstance(tour, str):
            yield StreamChunk(type="text", text=tour)
            self.final_emis_a.append(time.monotonic())
            yield StreamChunk(type="final", response=LLMResponse(
                content=tour, usage=TokenUsage(input_tokens=10, output_tokens=2)))
            return
        for call in tour:
            yield StreamChunk(type="tool_call", tool_call=call)
        if self.orphelin is not None:
            yield StreamChunk(type="tool_call", tool_call=self.orphelin)
        time.sleep(self.delai)                          # le modèle « parle encore »
        self.final_emis_a.append(time.monotonic())
        yield StreamChunk(type="final", response=LLMResponse(
            tool_calls=list(tour), usage=TokenUsage(input_tokens=10, output_tokens=2)))


class Vigie:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.demarrages: list[float] = []

    def top(self) -> None:
        with self.lock:
            self.demarrages.append(time.monotonic())


def _appel(i: int, nom: str = "lire") -> ToolCall:
    return ToolCall(id=f"c{i}", name=nom, arguments={"n": i})


def _agent(provider: Flux, vigie: Vigie, *, idempotent: bool, politique=None, trace=None) -> Agent:
    agent = Agent(provider, max_steps=4, tool_policy=politique, trace=trace)

    @agent.tool(idempotent=idempotent)
    def lire(n: int) -> str:
        """Lit un capteur."""
        vigie.top()
        return f"valeur {n}"

    return agent


def _run_stream(agent: Agent, prompt: str = "vas-y"):  # type: ignore[no-untyped-def]
    evenements = list(agent.run_stream(prompt))
    done = next(e for e in evenements if e.type == "done")
    return evenements, done


class TestRecouvrementReel:
    def test_l_outil_demarre_avant_le_final(self) -> None:
        vigie = Vigie()
        provider = Flux([[_appel(1), _appel(2)], "fini"], delai=0.25)
        _run_stream(_agent(provider, vigie, idempotent=True))
        final = provider.final_emis_a[0]
        assert len(vigie.demarrages) == 2
        assert all(d < final for d in vigie.demarrages), "aucun recouvrement : l'outil a attendu le final"

    def test_un_seul_appel_par_outil(self) -> None:
        vigie = Vigie()
        provider = Flux([[_appel(1), _appel(2)], "fini"])
        _, done = _run_stream(_agent(provider, vigie, idempotent=True))
        assert len(vigie.demarrages) == 2, "un résultat anticipé a été ré-exécuté"
        outils = [m for m in done.messages if m.role == "tool"]
        assert [m.tool_call_id for m in outils] == ["c1", "c2"]
        assert "valeur 1" in outils[0].content and "valeur 2" in outils[1].content


class TestSansIdempotentRienNePart:
    def test_l_outil_attend_le_final(self) -> None:
        vigie = Vigie()
        provider = Flux([[_appel(1)], "fini"], delai=0.2)
        _run_stream(_agent(provider, vigie, idempotent=False))
        final = provider.final_emis_a[0]
        assert len(vigie.demarrages) == 1
        assert vigie.demarrages[0] >= final, "un outil à effet de bord possible a été lancé en avance"

    def test_le_defaut_est_faux(self) -> None:
        agent = Agent(Flux(["x"]))

        @agent.tool
        def t() -> str:
            """t."""
            return "x"

        assert next(s for s in agent.registry.specs() if s.name == "t").idempotent is False


class TestLesGardesPassentAvant:
    def test_refuse_par_la_politique_pas_lance_en_avance(self) -> None:
        vigie = Vigie()
        provider = Flux([[_appel(1)], "fini"], delai=0.15)

        def politique(ctx):  # type: ignore[no-untyped-def]
            return "interdit" if ctx.name == "lire" else None

        _, done = _run_stream(_agent(provider, vigie, idempotent=True, politique=politique))
        assert vigie.demarrages == [], "la politique a été contournée par l'anticipation"
        outil = next(m for m in done.messages if m.role == "tool")
        assert "ToolPolicyDenied" in outil.content


class TestTranscriptIdentique:
    def test_memes_messages_et_memes_evenements_qu_en_voie_normale(self) -> None:
        avec, sans = Vigie(), Vigie()
        ev_avec, done_avec = _run_stream(_agent(Flux([[_appel(1), _appel(2)], "fini"]), avec, idempotent=True))
        ev_sans, done_sans = _run_stream(_agent(Flux([[_appel(1), _appel(2)], "fini"]), sans, idempotent=False))
        assert [(m.role, m.tool_call_id, m.content) for m in done_avec.messages] == \
               [(m.role, m.tool_call_id, m.content) for m in done_sans.messages]
        assert [(e.type, e.tool_name, e.tool_status) for e in ev_avec] == \
               [(e.type, e.tool_name, e.tool_status) for e in ev_sans]

    def test_la_trace_compte_les_anticipations_servies(self) -> None:
        fin: dict = {}
        types: list[str] = []

        def on_event(e) -> None:  # type: ignore[no-untyped-def]
            types.append(e.type)
            if e.type == "run_end":
                fin.update(e.payload)

        with TraceEmitter(on_event=on_event) as tr:
            _run_stream(_agent(Flux([[_appel(1), _appel(2)], "fini"]), Vigie(), idempotent=True, trace=tr))
        assert types.count("tool_call_early_start") == 2
        assert fin["early_tool_calls"] == 2

    def test_sans_anticipation_la_cle_est_absente(self) -> None:
        fin: dict = {}
        with TraceEmitter(on_event=lambda e: fin.update(e.payload) if e.type == "run_end" else None) as tr:
            _run_stream(_agent(Flux([[_appel(1)], "fini"]), Vigie(), idempotent=False, trace=tr))
        assert "early_tool_calls" not in fin


class TestOrphelin:
    def test_un_appel_anticipe_absent_du_final_est_jete(self) -> None:
        """Le flux annonce c9, le final ne le contient pas : le résultat est
        jeté, aucun message d'outil parasite, et le run se termine."""
        vigie = Vigie()
        provider = Flux([[_appel(1)], "fini"], orphelin=_appel(9))
        _, done = _run_stream(_agent(provider, vigie, idempotent=True))
        ids = [m.tool_call_id for m in done.messages if m.role == "tool"]
        assert ids == ["c1"]
        assert len(vigie.demarrages) == 2, "c9 a bien été lancé (idempotent, sans dégât)…"
        assert done.output == "fini"


class TestAnticipeEtReelMemeVerdict:
    def test_la_garde_anti_boucle_compte_l_appel_en_attente(self) -> None:
        """Sémantique existante de la garde : elle compte TOUS les appels du
        message assistant, donc deux appels identiques dans le MÊME tour voient
        chacun `seen=2` — avec `max_repeated_tool_calls=1`, les deux sont refusés.
        Avant 0.21.0 la vérification anticipée ne voyait ni l'appel courant ni
        son frère déjà lancé : les deux partaient en avance pour être refusés
        ensuite. Désormais elle compte l'appel en attente ET les frères du tour :
        seul le premier part (le second n'était pas encore connu), le second est
        retenu — et la vérification réelle rend le même verdict qu'avant."""
        vigie = Vigie()
        identiques = [ToolCall(id="c1", name="lire", arguments={"n": 7}),
                      ToolCall(id="c2", name="lire", arguments={"n": 7})]
        provider = Flux([identiques, "fini"], delai=0.15)
        agent = _agent(provider, vigie, idempotent=True)
        agent.max_repeated_tool_calls = 1
        _, done = _run_stream(agent)
        outils = [m for m in done.messages if m.role == "tool"]
        assert all("RepeatedCall" in m.content for m in outils), "sémantique du tour : les deux sont refusés"
        assert len(vigie.demarrages) == 1, "le second appel identique a été lancé en avance malgré son frère"
