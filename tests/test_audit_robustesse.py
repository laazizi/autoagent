"""Audit 0.22.0, lots 2 et 3 — robustesse, données, comptabilité.

Chaque test reproduit un défaut PROUVÉ sur la 0.21.0 (sans réseau ni clé) ; les
contre-épreuves vérifient que l'usage normal ne bouge pas. Aucun réglage par
défaut n'a changé : les options sûres sont opt-in, signalées par un
avertissement unique dans le journal.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import autoagent.logging as journal
from autoagent import (
    Agent,
    ApprovalRequired,
    DynamicToolBuilder,
    FactMemory,
    LLMRequest,
    LLMResponse,
    Message,
    ModelConfig,
    ProjectWorkspace,
    RecordSession,
    ReplayMismatch,
    ReplaySession,
    StreamChunk,
    SummarizingMemory,
    TokenUsage,
    ToolCall,
    TraceEmitter,
    delegate_to,
)
from autoagent.orchestrator import Orchestrator, Step
from autoagent.pipeline import PipelineError, PipelineManager
from autoagent.providers import anthropic as anthropic_mod
from autoagent.providers import gemini as gemini_mod
from autoagent.providers import openai as openai_mod
from autoagent.providers.anthropic import AnthropicProvider
from autoagent.providers.gemini import GeminiProvider
from autoagent.providers.openai import OpenAIProvider

U = TokenUsage


class Script:
    """Séquence de réponses (chaîne = texte, liste = appels) ; streaming natif."""

    def __init__(self, etapes: list, usage: int = 110) -> None:
        self.etapes = list(etapes)
        self.usage = usage

    def complete(self, request):
        e = self.etapes.pop(0) if self.etapes else "fin"
        u = U(input_tokens=self.usage - 10, output_tokens=10)
        if isinstance(e, LLMResponse):
            return e
        if isinstance(e, str):
            return LLMResponse(content=e, usage=u)
        return LLMResponse(content="", tool_calls=e, usage=u)

    def stream(self, request):
        rep = self.complete(request)
        for c in rep.tool_calls:
            yield StreamChunk(type="tool_call", tool_call=c)
        if rep.content:
            yield StreamChunk(type="text", text=rep.content)
        yield StreamChunk(type="final", response=rep)


def _appel(nom: str, cid: str, **args) -> list[ToolCall]:
    return [ToolCall(id=cid, name=nom, arguments=args)]


@pytest.fixture
def avertissements_neufs(monkeypatch):
    """`warn_once` se souvient par processus : chaque test repart de zéro."""
    monkeypatch.setattr(journal, "_DEJA_AVERTI", set())


# ── avertissements uniques sur les réglages historiques risqués ─────────────

class TestAvertissements:
    def test_evolution_sans_choix_explicite_avertit_une_fois(self, tmp_path, caplog, avertissements_neufs):
        import sys

        from autoagent import EvolutionRuntime
        rt = EvolutionRuntime(tmp_path, validation_command=[sys.executable, "-c", "pass"])
        with caplog.at_level(logging.WARNING, logger="autoagent"):
            rt.run_validation()
            rt.run_validation()
        assert sum("WHOLE host environment" in r.message for r in caplog.records) == 1

    @pytest.mark.parametrize("choix", [True, False])
    def test_un_choix_explicite_ne_declenche_rien(self, tmp_path, caplog, avertissements_neufs, choix):
        import sys

        from autoagent import EvolutionRuntime
        rt = EvolutionRuntime(tmp_path, validation_command=[sys.executable, "-c", "pass"], inherit_env=choix)
        with caplog.at_level(logging.WARNING, logger="autoagent"):
            rt.run_validation()
        assert not any("WHOLE host environment" in r.message for r in caplog.records)

    def test_outil_genere_sans_plafond_avertit(self, tmp_path, caplog, avertissements_neufs):
        payload = {"tool": {"name": "o", "description": "d", "input_schema": {"type": "object", "properties": {}},
                            "permissions": ["network"]},
                   "code": "def run(args, context):\n    return {}\n", "self_tests": []}

        class C:
            def complete(self, r):
                return LLMResponse(content=json.dumps(payload))

        from autoagent import ToolBuildRequest
        with caplog.at_level(logging.WARNING, logger="autoagent"):
            DynamicToolBuilder(C(), tools_dir=tmp_path).build(ToolBuildRequest(capability="x"))
        assert any("no host ceiling" in r.message for r in caplog.records)


# ── FactMemory : aucune écriture ne détruit plus le magasin ─────────────────

class TestMagasinDeFaits:
    def test_une_ecriture_interrompue_laisse_l_ancien_fichier_entier(self, tmp_path, monkeypatch):
        f = tmp_path / "faits.json"
        m = FactMemory(Script([]), path=f)
        for i in range(5):
            m.remember(f"fait important {i}")
        avant = f.read_text(encoding="utf-8")

        import autoagent._fichiers as fichiers

        def coupure(*a, **k):
            raise OSError("disque plein au moment du remplacement")

        monkeypatch.setattr(fichiers.os, "replace", coupure)
        m.remember("un sixième fait")                      # l'écriture échoue en route
        assert f.read_text(encoding="utf-8") == avant      # l'ancien fichier est intact
        assert len(json.loads(avant)["facts"]) == 5
        assert not list(tmp_path.glob(".faits.json.*.tmp"))  # pas de débris

    def test_un_fichier_illisible_est_mis_de_cote_jamais_ecrase(self, tmp_path):
        f = tmp_path / "faits.json"
        abime = '{"facts": [{"id": 1, "fact": "Mme X rappeler le soir"'   # tronqué
        f.write_text(abime, encoding="utf-8")
        m = FactMemory(Script([]), path=f)
        assert m.facts() == []
        m.remember("nouveau fait")
        copies = list(tmp_path.glob("faits.json.corrompu-*"))
        assert len(copies) == 1 and copies[0].read_text(encoding="utf-8") == abime

    def test_la_racine_doit_etre_un_objet(self, tmp_path):
        f = tmp_path / "faits.json"
        f.write_text("[1, 2, 3]", encoding="utf-8")
        assert FactMemory(Script([]), path=f).facts() == []   # plus d'AttributeError au démarrage


# ── SummarizingMemory : deux conversations ne se mélangent plus ────────────

class _Resumeur:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, request):
        self.prompts.append(request.messages[-1].content)
        return LLMResponse(content=f"RESUME#{len(self.prompts)}")


def _conv(prefixe: str, n: int) -> list[Message]:
    return [Message(role="user" if i % 2 == 0 else "assistant", content=f"{prefixe}{i}") for i in range(n)]


def test_summarizing_ne_melange_plus_deux_conversations() -> None:
    res = _Resumeur()
    sm = SummarizingMemory(res, max_messages=4, keep_recent=2)
    sm.compact(_conv("A-DUPONT-", 8))
    sm.compact(_conv("B", 9))
    assert "DUPONT" not in res.prompts[-1] and "RESUME#1" not in res.prompts[-1]
    assert all(f"B{i}" in res.prompts[-1] for i in range(8))       # rien de B n'est perdu
    assert sm.recall("DUPONT") == []                                 # l'archive de A est partie


def test_summarizing_continue_la_meme_conversation() -> None:
    """Contre-épreuve : la même conversation qui s'allonge reste incrémentale."""
    res = _Resumeur()
    sm = SummarizingMemory(res, max_messages=4, keep_recent=2)
    conv = _conv("A", 8)
    sm.compact(conv)
    sm.compact(conv + _conv("suite", 4))
    assert "RESUME#1" in res.prompts[-1]                             # résumé précédent repris
    assert "A0" not in res.prompts[-1]                               # déjà résumé, pas renvoyé


# ── Gemini : types nullables ───────────────────────────────────────────────

def test_les_types_optionnels_sont_traduits_pour_gemini() -> None:
    agent = Agent(Script([]))

    @agent.tool
    def chercher(ville: str, limite: int | None = None, mode: str | int | None = None,
                 tags: list[str] | None = None) -> dict:
        """x"""
        return {}

    spec = agent.registry.specs()[0]
    gemini = spec.as_gemini_declaration()["parameters"]["properties"]
    assert gemini["limite"] == {"type": "integer", "nullable": True}
    assert gemini["tags"] == {"type": "array", "items": {"type": "string"}, "nullable": True}
    assert gemini["mode"] == {"nullable": True, "anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert not any(isinstance(p.get("type"), list) for p in gemini.values())
    # Contre-épreuve : OpenAI / Anthropic reçoivent le JSON Schema d'origine.
    assert spec.input_schema["properties"]["limite"] == {"type": ["integer", "null"]}


# ── raison d'arrêt ──────────────────────────────────────────────────────────

def _gemini(monkeypatch, *reponses) -> Agent:
    file = list(reponses)
    monkeypatch.setattr(gemini_mod, "post_json", lambda *a, **k: file.pop(0))
    return Agent(GeminiProvider(ModelConfig(provider="gemini", model="m", api_key="k")))


class TestRaisonDArret:
    def test_un_appel_mal_forme_est_relance_une_fois(self, monkeypatch) -> None:
        evts: list = []
        agent = _gemini(monkeypatch,
                        {"candidates": [{"finishReason": "MALFORMED_FUNCTION_CALL"}]},
                        {"candidates": [{"content": {"parts": [{"text": "RDV noté jeudi 18h"}]},
                                         "finishReason": "STOP"}]})
        agent.trace = TraceEmitter(on_event=evts.append)
        resultat = agent.run("Prends le rendez-vous.")
        assert resultat.output == "RDV noté jeudi 18h" and resultat.finish_reason == "stop"
        assert [e.type for e in evts].count("llm_retry") == 1

    def test_la_relance_n_a_lieu_qu_une_fois(self, monkeypatch) -> None:
        mal = {"candidates": [{"finishReason": "MALFORMED_FUNCTION_CALL"}]}
        resultat = _gemini(monkeypatch, mal, mal).run("x")
        assert resultat.output == "" and resultat.finish_reason == "malformed"

    def test_un_blocage_securite_est_dit_et_pas_relance(self, monkeypatch) -> None:
        agent = _gemini(monkeypatch, {"candidates": [{"finishReason": "SAFETY"}]})
        resultat = agent.run("x")
        assert resultat.output == "" and resultat.finish_reason == "content_filter"
        assert resultat.steps == 1

    def test_un_prompt_bloque_en_entree(self, monkeypatch) -> None:
        resultat = _gemini(monkeypatch, {"promptFeedback": {"blockReason": "SAFETY"}}).run("x")
        assert resultat.finish_reason == "content_filter"

    def test_openai_rapporte_la_coupure(self, monkeypatch) -> None:
        monkeypatch.setattr(openai_mod, "post_json", lambda *a, **k: {
            "choices": [{"message": {"content": "début d'une réponse coup"}, "finish_reason": "length"}]})
        p = OpenAIProvider(ModelConfig(provider="openai", model="m", api_key="k"))
        assert p.complete(LLMRequest(messages=[Message(role="user", content="x")])).finish_reason == "length"

    def test_openai_sans_choix_donne_une_erreur_lisible(self, monkeypatch) -> None:
        from autoagent import ProviderError
        monkeypatch.setattr(openai_mod, "post_json", lambda *a, **k: {"choices": []})
        p = OpenAIProvider(ModelConfig(provider="openai", model="m", api_key="k"))
        with pytest.raises(ProviderError, match="no choices"):
            p.complete(LLMRequest(messages=[Message(role="user", content="x")]))

    def test_anthropic_ne_vide_plus_des_arguments_coupes(self, monkeypatch) -> None:
        evts = [{"type": "message_start", "message": {"model": "c", "usage": {"input_tokens": 5}}},
                {"type": "content_block_start", "index": 0,
                 "content_block": {"type": "tool_use", "id": "t1", "name": "purger"}},
                {"type": "content_block_delta", "index": 0,
                 "delta": {"type": "input_json_delta", "partial_json": '{"filtre": "dossier_45'}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 9}}]
        monkeypatch.setattr(anthropic_mod, "post_sse", lambda *a, **k: iter(evts))
        p = AnthropicProvider(ModelConfig(provider="anthropic", model="c", api_key="k"))
        final = list(p.stream(LLMRequest(messages=[Message(role="user", content="x")])))[-1].response
        assert final.tool_calls[0].arguments == {"_raw": '{"filtre": "dossier_45'}
        assert final.finish_reason == "length"


# ── streaming : l'état reprenable est transmis ─────────────────────────────

class TestEtatEnStreaming:
    def _erreur(self, agent: Agent, **k):
        return [e for e in agent.run_stream("go", **k) if e.type == "error"][-1]

    def test_plafond_d_etapes(self) -> None:
        agent = Agent(Script([_appel("ping", f"c{i}") for i in range(9)]), max_steps=2)
        agent.tool(lambda: {"ok": 1}, name="ping")
        err = self._erreur(agent)
        assert err.state is not None and err.state.step == 2
        agent.max_steps = 3
        with pytest.raises(Exception, match="max_steps=3"):    # repris là où il était : étape 3
            agent.resume(err.state)

    def test_budget_et_reprise_en_streaming(self) -> None:
        agent = Agent(Script([_appel("ping", "c1"), _appel("ping", "c2"), "fini"]), token_budget=150)
        agent.tool(lambda: {"ok": 1}, name="ping")
        err = self._erreur(agent)
        assert err.state is not None
        agent.token_budget = 10_000
        assert agent.resume(err.state).output == "fini"

    def test_annulation(self) -> None:
        stop = threading.Event()
        stop.set()
        err = self._erreur(Agent(Script(["x"])), cancel_token=stop)
        assert err.error == "cancelled" and err.state is not None


# ── exécution anticipée : une seule vérification officielle ───────────────

class TestExecutionAnticipee:
    def test_les_refus_sont_traces_une_fois(self) -> None:
        evts: list = []
        agent = Agent(Script([_appel("lire", "c1", x=1), _appel("lire", "c2", x=1), _appel("lire", "c3", x=1), "fin"]),
                      max_repeated_tool_calls=1, trace=TraceEmitter(on_event=evts.append), max_steps=5)
        agent.tool(lambda x: {"v": x}, name="lire", idempotent=True)
        list(agent.run_stream("go"))
        assert [e.type for e in evts].count("loop_guard_block") == 2

    def test_le_mode_temoin_compte_juste(self) -> None:
        evts: list = []
        agent = Agent(Script([_appel("lire", "c1", x=1), _appel("lire", "c2", x=1), _appel("lire", "c3", x=1), "fin"]),
                      max_repeated_tool_calls=1, shadow_guards=True,
                      trace=TraceEmitter(on_event=evts.append), max_steps=5)
        agent.tool(lambda x: {"v": x}, name="lire", idempotent=True)
        list(agent.run_stream("go"))
        assert next(e for e in evts if e.type == "run_end").payload["would_block"] == 2

    def test_la_politique_de_l_hote_n_est_appelee_qu_une_fois(self) -> None:
        vus: list[str] = []
        agent = Agent(Script([_appel("lire", "c1", x=1), "fin"]), tool_policy=lambda ctx: vus.append(ctx.call.id))
        agent.tool(lambda x: {"v": x}, name="lire", idempotent=True)
        list(agent.run_stream("go"))
        assert vus == ["c1"]

    def test_sans_refus_l_outil_anticipe_tourne_une_fois(self) -> None:
        lus: list[int] = []
        agent = Agent(Script([_appel("lire", "c1", x=3), "fin"]), tool_policy=lambda ctx: None)
        _lecteur(agent, lus)
        list(agent.run_stream("go"))
        assert lus == [3]

    def test_un_refus_de_la_politique_reste_un_refus(self) -> None:
        """Contre-épreuve : le verdict mémorisé s'applique bien au tour."""
        lus: list[int] = []
        agent = Agent(Script([_appel("lire", "c1", x=1), "fin"]), tool_policy=lambda ctx: "non")
        _lecteur(agent, lus)
        evs = list(agent.run_stream("go"))
        assert lus == [] and any(e.type == "tool_end" and e.tool_status == "error" for e in evs)

    def test_une_approbation_en_plein_flux_garde_l_appel_en_attente(self) -> None:
        decision: dict[str, bool] = {}

        def politique(ctx):
            if decision.get(ctx.call.id):
                return None
            raise ApprovalRequired("ok ?")

        lus: list[int] = []
        agent = Agent(Script([_appel("lire", "c1", x=7), "fin"]), tool_policy=politique)
        _lecteur(agent, lus)
        err = [e for e in agent.run_stream("go") if e.type == "error"][-1]
        assert err.state.messages[-1].tool_calls[0].id == "c1"       # l'appel attend son verdict
        assert lus == []                                              # rien n'est parti en avance
        decision["c1"] = True
        assert agent.resume(err.state).output == "fin" and lus == [7]


# ── Orchestrator : les callbacks de l'hôte ne cassent plus un appel ────────

class _Flux:
    def __init__(self, interpret: str) -> None:
        self.config = ModelConfig(provider="fake", model="fake")
        self.interpret = interpret
        self.contextes: list[dict[str, Any]] = []

    def complete(self, request):
        return LLMResponse(content=self.interpret)

    def stream(self, request):
        self.contextes.append(json.loads(request.messages[1].content))
        yield StreamChunk(type="text", text="?")
        yield StreamChunk(type="final", response=LLMResponse(content="?"))


def _orchestrateur(flux: _Flux, **k) -> Orchestrator:
    reponses: dict[str, Any] = {}
    return Orchestrator(flux, current_steps=lambda: [Step("age", {"ask": "age"})] if "age" not in reponses else [],
                        record=k.pop("record", lambda sid, v: reponses.__setitem__(sid, v)), **k)


class TestOrchestrator:
    def test_le_texte_d_une_exception_n_atteint_pas_la_personne(self) -> None:
        flux = _Flux('{"status": "answered", "values": [{"id": "age", "value": 30}]}')

        def record(sid, v):
            raise RuntimeError("connection to server at 10.0.0.5 failed: password authentication")

        list(_orchestrateur(flux, record=record).turn("j'ai 30 ans"))
        erreur = flux.contextes[-1]["validation_error"]
        assert "10.0.0.5" not in erreur and "password" not in erreur

    def test_describe_qui_plante_n_annule_pas_l_enregistrement(self) -> None:
        flux = _Flux('{"status": "answered", "values": [{"id": "age", "value": 30}]}')
        evs = list(_orchestrateur(flux, describe=lambda sid, v: 1 / 0).turn("30"))
        assert any(e.type == "recorded" for e in evs)

    def test_accept_extra_qui_plante_vaut_refus(self) -> None:
        flux = _Flux('{"status": "answered", "values": [{"id": "autre", "value": 1}]}')
        evs = list(_orchestrateur(flux, accept_extra=lambda sid: 1 / 0).turn("x"))
        assert not any(e.type == "recorded" for e in evs)


# ── comptabilité : la dépense des échecs et des tâches de fond ─────────────

def _lecteur(agent: Agent, journal: list[int]) -> None:
    """Outil idempotent ANNOTÉ : une lambda est typée « string » et refuserait l'entier."""
    @agent.tool(idempotent=True)
    def lire(x: int) -> dict:
        """Lit une valeur."""
        journal.append(x)
        return {"v": x}


def _boucle(max_steps: int = 5) -> Agent:
    a = Agent(Script([_appel("ping", f"s{i}") for i in range(20)]), max_steps=max_steps)
    a.tool(lambda: {"ok": 1}, name="ping")
    return a


def test_as_tool_compte_la_depense_d_un_sous_agent_en_echec() -> None:
    parent = Agent(Script([_appel("spe", "p1", request="x"), "fin"]))
    parent.add_tool(_boucle().as_tool(name="spe", description="d"))
    assert parent.run("go").usage.total_tokens == 2 * 110 + 5 * 110


def test_delegate_to_compte_la_depense_d_un_specialiste_en_echec() -> None:
    parent = Agent(Script([_appel("delegate", "p1", requests=[{"specialist": "b", "request": "x"}]), "fin"]))
    parent.add_tool(delegate_to({"b": _boucle()}))
    assert parent.run("go").usage.total_tokens == 2 * 110 + 5 * 110


def test_deux_appels_paralleles_du_meme_specialiste_sont_comptes() -> None:
    """Canal PAR THREAD : avant, l'attribut partagé perdait une des deux dépenses.

    La course est rendue DÉTERMINISTE : une barrière dans le registre attend que
    les DEUX appels aient fini (donc posé leur dépense) avant que l'un ou l'autre
    ne la lise. Avec un canal partagé, le second écrase le premier."""
    from autoagent import ToolRegistry

    barriere = threading.Barrier(2, timeout=10)

    class RegistreSynchronise(ToolRegistry):
        def execute(self, call, context=None):
            resultat = super().execute(call, context=context)
            barriere.wait()
            return resultat

    class Spe(Script):
        def complete(self, request):
            return LLMResponse(content="ok", usage=U(input_tokens=90, output_tokens=10))

    parent = Agent(Script([[ToolCall(id="a", name="spe", arguments={"request": "1"}),
                            ToolCall(id="b", name="spe", arguments={"request": "2"})], "fin"]),
                   parallel_tool_calls=True, registry=RegistreSynchronise())
    parent.add_tool(Agent(Spe([])).as_tool(name="spe", description="d"))
    assert parent.run("go").usage.total_tokens == 2 * 110 + 2 * 100


def test_la_consolidation_en_arriere_plan_est_comptee() -> None:
    class Extraction:
        def __init__(self):
            self.n = 0

        def complete(self, request):
            self.n += 1
            time.sleep(0.2)
            return LLMResponse(content='{"operations": []}', usage=U(input_tokens=1000, output_tokens=200))

    ext = Extraction()
    mem = FactMemory(ext, max_messages=4, keep_recent=2, background=True)
    hist = _conv("m", 10)
    mem.compact(hist)                                   # lance l'extraction, rend la main tout de suite
    mem.flush()                                         # l'extraction finit APRÈS
    mem.compact([*hist, Message(role="user", content="suite")])
    assert mem.last_usage is not None and mem.last_usage.total_tokens >= 1200 * ext.n - 1200


# ── qualité ────────────────────────────────────────────────────────────────

def test_read_file_reste_identique_en_lisant_borne(tmp_path: Path) -> None:
    ws = ProjectWorkspace(tmp_path, max_read_chars=10)
    (tmp_path / "petit.txt").write_text("abc", encoding="utf-8")
    (tmp_path / "gros.txt").write_text("é" * 5000, encoding="utf-8")
    assert ws.read_file("petit.txt") == {"path": "petit.txt", "content": "abc", "truncated": False, "chars": 3}
    gros = ws.read_file("gros.txt")
    assert gros["content"] == "é" * 10 and gros["truncated"] and gros["chars"] == 5000


def test_un_slot_de_pipeline_ne_designe_que_des_modules_autorises(tmp_path: Path) -> None:
    pm = PipelineManager(ProjectWorkspace(tmp_path), allowed_module_prefixes=("plugins.",))
    with pytest.raises(PipelineError, match="not allowed"):
        pm.replace_slot("etape", "subprocess", callable_name="run")
    assert pm.replace_slot("etape", "plugins.doubler")["ok"]
    assert PipelineManager(ProjectWorkspace(tmp_path)).replace_slot("x", "subprocess")["ok"]  # défaut inchangé


def test_le_decorateur_tool_accepte_idempotent() -> None:
    from autoagent import tool

    @tool(idempotent=True)
    def lire(x: int) -> dict:
        """x"""
        return {}

    assert lire.__autoagent_tool_spec__.idempotent


def test_le_rejeu_peut_verifier_les_prompts(tmp_path: Path) -> None:
    fixture = tmp_path / "f.jsonl"
    with RecordSession(fixture) as rec:
        Agent(rec.provider(Script(["ok"])), registry=rec.registry(), system_prompt="Version 1").run("go")
    for strict_prompts, attendu in ((False, "ok"), (True, None)):
        with ReplaySession(fixture, check_prompts=strict_prompts) as rep:
            agent = Agent(rep.provider(), registry=rep.registry(), system_prompt="Version 2 modifiée")
            if attendu is None:
                with pytest.raises(ReplayMismatch, match="CONTENU"):
                    agent.run("go")
            else:
                assert agent.run("go").output == attendu


def test_un_ancien_fixture_sans_empreinte_se_rejoue(tmp_path: Path) -> None:
    fixture = tmp_path / "ancien.jsonl"
    fixture.write_text(json.dumps({"kind": "llm", "channel": "agent", "seq": 1,
                                   "request": {"messages": 2, "last_user": "go", "tools": []},
                                   "response": {"content": "ok"}}) + "\n", encoding="utf-8")
    with ReplaySession(fixture, check_prompts=True) as rep:
        assert Agent(rep.provider(), registry=rep.registry()).run("go").output == "ok"
