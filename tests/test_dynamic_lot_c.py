"""Outils dynamiques — lot C (0.22.0) : plus de pouvoir, sans changer les défauts.

* fonctions de l'HÔTE appelables par un outil généré (`context["call_host"]`) ;
* `run_python` : un extrait ÉPHÉMÈRE, validé, sandboxé, jamais gardé (opt-in) ;
* sandbox « chaud » : un worker persistant par outil au lieu d'un processus par
  appel — mesuré ~107 ms/appel à froid, ~6 ms à chaud (opt-in, avec ses
  contreparties dites).

Défauts : `host_functions=None`, aucun `run_python`, `warm=False`.
"""

from __future__ import annotations

import json
import tempfile
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from autoagent import Agent, DynamicToolBuilder, PythonRunner, ToolBuildRequest
from autoagent.errors import ToolError, ToolValidationError
from autoagent.providers.base import LLMProvider
from autoagent.sandbox import SubprocessSandbox, load_generated_tool
from autoagent.schema import LLMRequest, LLMResponse, ModelConfig, TokenUsage, ToolCall


class Sequence(LLMProvider):
    def __init__(self, contenus: list[str]) -> None:
        super().__init__(ModelConfig(provider="f", model="f", api_key="x"))
        self.contenus = list(contenus)
        self.requetes: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requetes.append(request)
        return LLMResponse(content=self.contenus.pop(0), model="f")


def _payload(code: str, tests: Any = None, nom: str = "outil") -> str:
    return json.dumps({
        "tool": {"name": nom, "description": "d",
                 "input_schema": {"type": "object", "properties": {"cle": {"type": "string"}}}, "permissions": []},
        "code": code, "self_tests": [] if tests is None else tests,
    })


# ── Fonctions de l'hôte ─────────────────────────────────────────────────────

CODE_HOTE = "def run(args, context):\n    return {'v': context['call_host']('lire', {'cle': args['cle']})}\n"


class _Hote:
    def __init__(self) -> None:
        self.appels: list[str] = []

    def lire(self, cle: str) -> dict[str, str]:
        """Lit une valeur en lecture seule."""
        self.appels.append(cle)
        return {"valeur": cle.upper()}


class TestFonctionsDeLHote:
    def test_par_defaut_rien_n_est_expose(self, tmp_path: Path) -> None:
        b = DynamicToolBuilder(Sequence([_payload(CODE_HOTE)]), tools_dir=tmp_path)
        assert b.host_functions is None and b._host_prompt() == ""
        assert "call_host" not in b._system_prompt()

    def test_le_constructeur_voit_noms_signatures_et_docstring(self, tmp_path: Path) -> None:
        hote = _Hote()
        b = DynamicToolBuilder(Sequence([]), tools_dir=tmp_path, host_functions={"lire": hote.lire})
        prompt = b._system_prompt()
        assert "context['call_host'](name" in prompt
        assert "- lire(cle: 'str') -> 'dict[str, str]': Lit une valeur en lecture seule." in prompt
        assert "empty self_tests list" in prompt

    def test_un_outil_construit_appelle_la_fonction_de_l_hote(self, tmp_path: Path) -> None:
        hote = _Hote()
        b = DynamicToolBuilder(Sequence([_payload(CODE_HOTE)]), tools_dir=tmp_path, host_functions={"lire": hote.lire})
        outil = b.build(ToolBuildRequest(capability="x"))
        assert hote.appels == [], "le build ne touche jamais l'hôte"
        assert outil(cle="abc") == {"v": {"valeur": "ABC"}}
        assert hote.appels == ["abc"]

    def test_les_self_tests_tournent_sans_l_hote(self, tmp_path: Path) -> None:
        hote = _Hote()
        avec_test = _payload(CODE_HOTE, tests=[{"args": {"cle": "a"}, "expect_contains": "A"}])
        b = DynamicToolBuilder(Sequence([avec_test]), tools_dir=tmp_path, host_functions={"lire": hote.lire})
        with pytest.raises((ToolError, ToolValidationError)):
            b.build(ToolBuildRequest(capability="x"))
        assert hote.appels == [], "un self-test n'a aucun effet de bord sur l'hôte"
        assert not list(tmp_path.glob("*.py"))

    def test_un_nom_hors_liste_blanche_est_refuse(self, tmp_path: Path) -> None:
        hote = _Hote()
        code = "def run(args, context):\n    return context['call_host']('supprimer_tout', {})\n"
        b = DynamicToolBuilder(Sequence([_payload(code)]), tools_dir=tmp_path, host_functions={"lire": hote.lire})
        outil = b.build(ToolBuildRequest(capability="x"))
        with pytest.raises(ToolError, match="host function not allowed: supprimer_tout"):
            outil(cle="a")

    def test_une_exception_de_l_hote_devient_une_erreur_lisible(self, tmp_path: Path) -> None:
        def casse(cle: str) -> None:
            raise PermissionError("accès refusé")

        code = CODE_HOTE.replace("'lire'", "'casse'")
        outil = DynamicToolBuilder(Sequence([_payload(code)]), tools_dir=tmp_path, host_functions={"casse": casse}).build(
            ToolBuildRequest(capability="x"))
        with pytest.raises(ToolError, match="PermissionError: accès refusé"):
            outil(cle="a")

    def test_le_mode_normal_n_est_pas_touche_sans_fonctions(self, tmp_path: Path) -> None:
        code = "def run(args, context):\n    return {'ok': 'call_host' not in context}\n"
        outil = DynamicToolBuilder(Sequence([_payload(code)]), tools_dir=tmp_path).build(ToolBuildRequest(capability="x"))
        assert outil.host_functions is None and outil(cle="a") == {"ok": True}

    def test_un_appel_peut_surcharger_les_fonctions_de_l_outil(self, tmp_path: Path) -> None:
        un, deux = _Hote(), _Hote()
        outil = DynamicToolBuilder(Sequence([_payload(CODE_HOTE)]), tools_dir=tmp_path,
                                   host_functions={"lire": un.lire}).build(ToolBuildRequest(capability="x"))
        outil(cle="x", host_functions={"lire": deux.lire})
        assert un.appels == [] and deux.appels == ["x"]

    def test_la_bibliotheque_rattache_aussi_les_fonctions(self, tmp_path: Path) -> None:
        DynamicToolBuilder(Sequence([_payload(CODE_HOTE)]), tools_dir=tmp_path, persist=True).build(
            ToolBuildRequest(capability="x"))
        hote = _Hote()
        neuf = DynamicToolBuilder(Sequence([]), tools_dir=tmp_path, persist=True, host_functions={"lire": hote.lire})
        (outil,) = neuf.load_library()
        assert outil(cle="z") == {"v": {"valeur": "Z"}}

    def test_enable_dynamic_tools_accepte_les_fonctions_et_un_run_complet_les_appelle(self, tmp_path: Path) -> None:
        hote = _Hote()
        builder = DynamicToolBuilder(Sequence([_payload(CODE_HOTE, nom="lecteur")]), tools_dir=tmp_path)

        class Principal(LLMProvider):
            def __init__(self) -> None:
                super().__init__(ModelConfig(provider="p", model="p", api_key="x"))
                self.reponses = [
                    LLMResponse(tool_calls=[ToolCall(id="c1", name="create_python_tool",
                                                     arguments={"capability": "x", "tool_name": "lecteur"})]),
                    LLMResponse(tool_calls=[ToolCall(id="c2", name="lecteur", arguments={"cle": "k"})]),
                    LLMResponse(content="fini"),
                ]

            def complete(self, request: LLMRequest) -> LLMResponse:
                return self.reponses.pop(0)

        agent = Agent(Principal(), max_steps=6)
        agent.enable_dynamic_tools(builder, host_functions={"lire": hote.lire})
        assert builder.host_functions is not None
        resultat = agent.run("go")
        assert hote.appels == ["k"]
        assert any("K" in (m.content or "") for m in resultat.messages if m.role == "tool")


# ── run_python ──────────────────────────────────────────────────────────────

class _Principal(LLMProvider):
    def __init__(self, reponses: list[LLMResponse]) -> None:
        super().__init__(ModelConfig(provider="p", model="p", api_key="x"))
        self.reponses = list(reponses)

    def complete(self, request: LLMRequest) -> LLMResponse:
        return self.reponses.pop(0)


def _rp(code: str, **args: Any) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id="c1", name="run_python",
                                            arguments={"code": code, **({"args": args} if args else {})})],
                       usage=TokenUsage(input_tokens=10, output_tokens=5))


FIN = LLMResponse(content="fini", usage=TokenUsage(input_tokens=10, output_tokens=5))
CARRE = "def run(args, context):\n    return {'carre': args['n'] ** 2}\n"


class TestRunPython:
    def test_absent_par_defaut(self) -> None:
        assert "run_python" not in Agent(_Principal([FIN])).registry

    def test_s_active_explicitement(self) -> None:
        agent = Agent(_Principal([FIN]))
        agent.enable_run_python()
        assert "run_python" in agent.registry
        spec = next(s for s in agent.registry.specs() if s.name == "run_python")
        assert spec.input_schema["required"] == ["code"]

    def test_execute_un_extrait_et_rend_le_resultat(self) -> None:
        sortie = PythonRunner()(CARRE, {"n": 12})
        assert sortie == {"result": {"carre": 144}, "stdout": ""}

    def test_la_sortie_standard_est_rendue(self) -> None:
        code = "def run(args, context):\n    print('trace')\n    return 1\n"
        assert PythonRunner()(code) == {"result": 1, "stdout": "trace\n"}

    def test_dans_l_agent_le_modele_voit_le_resultat(self) -> None:
        agent = Agent(_Principal([_rp(CARRE, n=7), FIN]), max_steps=4)
        agent.enable_run_python()
        outil = next(m for m in agent.run("go").messages if m.role == "tool")
        assert "49" in outil.content

    def test_le_validateur_refuse_ce_qu_il_refuse_ailleurs(self) -> None:
        for code, motif in [
            ("def run(args, context):\n    return eval('1')\n", "eval"),
            ("import os\ndef run(args, context):\n    return 1\n", "os"),
            ("import socket\ndef run(args, context):\n    return 1\n", "network"),
            ("def run(args, context):\n    return open('x').read()\n", "filesystem"),
        ]:
            with pytest.raises(ToolValidationError, match=motif):
                PythonRunner()(code)

    def test_sans_run_est_refuse(self) -> None:
        with pytest.raises(ToolValidationError, match="run"):
            PythonRunner()("x = 1\n")

    def test_les_permissions_viennent_de_l_hote_jamais_du_modele(self) -> None:
        code = "import socket\ndef run(args, context):\n    return 'ok'\n"
        assert PythonRunner(permissions=["network"])(code)["result"] == "ok"

    def test_code_vide_trop_long_ou_args_invalides(self) -> None:
        with pytest.raises(ToolValidationError, match="empty"):
            PythonRunner()("  ")
        with pytest.raises(ToolValidationError, match="limit is 50"):
            PythonRunner(max_code_chars=50)(CARRE)
        with pytest.raises(ToolValidationError, match="args must be an object"):
            PythonRunner()(CARRE, ["n"])        # type: ignore[arg-type]

    def test_une_erreur_d_execution_est_rendue_au_modele_sans_planter_le_run(self) -> None:
        code = "def run(args, context):\n    return 1 / 0\n"
        agent = Agent(_Principal([_rp(code), FIN]), max_steps=4)
        agent.enable_run_python()
        resultat = agent.run("go")
        assert "ZeroDivisionError" in next(m for m in resultat.messages if m.role == "tool").content
        assert resultat.output == "fini"

    def test_un_extrait_qui_boucle_est_arrete(self) -> None:
        code = "def run(args, context):\n    while True:\n        pass\n"
        runner = PythonRunner(SubprocessSandbox(timeout=1.5))
        t0 = time.monotonic()
        with pytest.raises(ToolError, match="timed out"):
            runner(code)
        assert time.monotonic() - t0 < 8

    def test_rien_ne_reste_sur_le_disque(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        PythonRunner()(CARRE, {"n": 2})
        with pytest.raises(ToolError):
            PythonRunner()("def run(args, context):\n    return 1 / 0\n")
        assert list(tmp_path.iterdir()) == []

    def test_le_plafond_par_run_compte_aussi_les_echecs_et_repart_au_run_suivant(self) -> None:
        mauvais = "def run(args, context):\n    return 1 / 0\n"
        agent = Agent(_Principal([_rp(mauvais), _rp(CARRE, n=2), _rp(CARRE, n=3), FIN,
                                  _rp(CARRE, n=4), FIN]), max_steps=8)
        agent.enable_run_python(max_runs_per_run=2)
        premier = agent.run("go")
        textes = [m.content for m in premier.messages if m.role == "tool"]
        assert "ZeroDivisionError" in textes[0] and "4" in textes[1]
        assert "budget exhausted" in textes[2], "le 3e appel du run est refusé"
        second = agent.run("go")
        assert "16" in next(m for m in second.messages if m.role == "tool").content, "le compteur repart à zéro"

    def test_un_sandbox_chaud_est_remplace_par_un_froid(self) -> None:
        runner = PythonRunner(SubprocessSandbox(timeout=10, warm=True))
        assert runner.sandbox.warm is False

    def test_avertit_une_fois_hors_docker(self, caplog) -> None:
        import logging

        from autoagent import logging as journal
        journal._DEJA_AVERTI.discard("dynamic.run_python_not_docker")
        with caplog.at_level(logging.WARNING, logger="autoagent"):
            PythonRunner()
            PythonRunner()
        assert sum("run_python executes model-written code" in r.message for r in caplog.records) == 1


# ── Le sandbox chaud ────────────────────────────────────────────────────────

COMPTEUR = textwrap.dedent('''
    TOOL = {"name": "compteur", "description": "d", "input_schema": {"type": "object", "properties": {}}}
    COMPTE = 0

    def run(args, context):
        global COMPTE
        COMPTE += 1
        print("du bruit sur stdout qui ne doit pas casser le protocole")
        if args.get("boucle"):
            while True:
                pass
        if args.get("plante"):
            raise ValueError("planté exprès")
        if args.get("bizarre"):
            return {"objet": object}
        return {"n": COMPTE, "x2": args.get("x", 0) * 2}
''')


@pytest.fixture
def outil_chaud(tmp_path: Path):
    fichier = tmp_path / "compteur.py"
    fichier.write_text(COMPTEUR, encoding="utf-8")
    with SubprocessSandbox(timeout=10, warm=True) as bac:
        yield load_generated_tool(fichier, sandbox=bac), bac, fichier


class TestSandboxChaud:
    def test_par_defaut_c_est_froid_un_processus_par_appel(self, tmp_path: Path) -> None:
        fichier = tmp_path / "compteur.py"
        fichier.write_text(COMPTEUR, encoding="utf-8")
        bac = SubprocessSandbox(timeout=10)
        assert bac.warm is False
        outil = load_generated_tool(fichier, sandbox=bac)
        assert [outil()["n"] for _ in range(3)] == [1, 1, 1]
        assert bac._workers == {}

    def test_un_seul_worker_sert_tous_les_appels(self, outil_chaud) -> None:
        outil, bac, _ = outil_chaud
        assert [outil(x=i)["n"] for i in range(4)] == [1, 2, 3, 4]
        assert len(bac._workers) == 1

    def test_meme_resultat_qu_a_froid(self, outil_chaud, tmp_path: Path) -> None:
        outil, _bac, fichier = outil_chaud
        froid = load_generated_tool(fichier, sandbox=SubprocessSandbox(timeout=10))
        assert outil(x=21)["x2"] == froid(x=21)["x2"] == 42

    def test_print_dans_l_outil_ne_casse_pas_le_protocole(self, outil_chaud) -> None:
        outil, *_ = outil_chaud
        assert outil(x=1) == {"n": 1, "x2": 2}
        assert outil(x=2) == {"n": 2, "x2": 4}

    def test_une_exception_de_l_outil_ne_tue_pas_le_worker(self, outil_chaud) -> None:
        outil, *_ = outil_chaud
        outil()
        with pytest.raises(ToolError, match="ValueError: planté exprès"):
            outil(plante=True)
        assert outil()["n"] == 3, "même worker : l'état a continué (compteurs 1, 2 puis 3)"

    def test_un_resultat_non_serialisable_est_rendu_en_repr(self, outil_chaud) -> None:
        outil, *_ = outil_chaud
        assert "object" in outil(bizarre=True)["objet"]

    def test_un_delai_depasse_tue_le_worker_et_le_suivant_repart_a_neuf(self, tmp_path: Path) -> None:
        fichier = tmp_path / "compteur.py"
        fichier.write_text(COMPTEUR, encoding="utf-8")
        with SubprocessSandbox(timeout=1.5, warm=True) as bac:
            outil = load_generated_tool(fichier, sandbox=bac)
            assert outil()["n"] == 1
            ancien = next(iter(bac._workers.values())).proc
            with pytest.raises(ToolError, match="timed out"):
                outil(boucle=True)
            assert ancien.poll() is not None, "le processus qui bouclait est bien mort"
            assert outil()["n"] == 1, "un worker neuf : l'état de l'ancien n'a pas fui"

    def test_recyclage_apres_warm_max_calls(self, tmp_path: Path) -> None:
        fichier = tmp_path / "compteur.py"
        fichier.write_text(COMPTEUR, encoding="utf-8")
        with SubprocessSandbox(timeout=10, warm=True, warm_max_calls=3) as bac:
            outil = load_generated_tool(fichier, sandbox=bac)
            assert [outil()["n"] for _ in range(7)] == [1, 2, 3, 1, 2, 3, 1]

    def test_un_fichier_reecrit_n_est_pas_servi_par_l_ancien_worker(self, outil_chaud) -> None:
        outil, _bac, fichier = outil_chaud
        assert outil(x=2)["x2"] == 4
        fichier.write_text(COMPTEUR.replace('args.get("x", 0) * 2', 'args.get("x", 0) * 10'), encoding="utf-8")
        assert outil(x=2)["x2"] == 20

    def test_une_erreur_au_chargement_est_rendue_puis_le_worker_n_est_pas_garde(self, tmp_path: Path) -> None:
        fichier = tmp_path / "casse.py"
        fichier.write_text(
            'TOOL = {"name": "casse", "description": "d", "input_schema": {"type": "object"}}\n'
            "raise RuntimeError('échec à l import')\n"
            "def run(args, context):\n    return 1\n", encoding="utf-8")
        with SubprocessSandbox(timeout=10, warm=True) as bac:
            outil = load_generated_tool(fichier, sandbox=bac)
            for _ in range(2):
                with pytest.raises(ToolError, match="failed to load: RuntimeError"):
                    outil()

    def test_plafond_de_workers_vivants(self, tmp_path: Path) -> None:
        with SubprocessSandbox(timeout=10, warm=True, warm_max_workers=2) as bac:
            outils = []
            for i in range(3):
                f = tmp_path / f"t{i}.py"
                f.write_text(COMPTEUR.replace('"compteur"', f'"t{i}"'), encoding="utf-8")
                outils.append(load_generated_tool(f, sandbox=bac))
            procs = []
            for o in outils:
                o()
                procs.append(list(bac._workers.values())[-1].proc)
            assert len(bac._workers) == 2
            assert procs[0].poll() is not None, "le plus ancien a été fermé"
            assert procs[2].poll() is None

    def test_close_ferme_les_processus(self, tmp_path: Path) -> None:
        fichier = tmp_path / "compteur.py"
        fichier.write_text(COMPTEUR, encoding="utf-8")
        bac = SubprocessSandbox(timeout=10, warm=True)
        outil = load_generated_tool(fichier, sandbox=bac)
        outil()
        proc = next(iter(bac._workers.values())).proc
        assert proc.poll() is None
        bac.close()
        assert proc.poll() is not None and bac._workers == {}

    def test_appels_concurrents_sur_un_meme_outil(self, outil_chaud) -> None:
        outil, *_ = outil_chaud
        resultats: list[int] = []
        verrou = threading.Lock()

        def tache() -> None:
            for _ in range(5):
                n = outil()["n"]
                with verrou:
                    resultats.append(n)

        fils = [threading.Thread(target=tache) for _ in range(4)]
        for f in fils:
            f.start()
        for f in fils:
            f.join()
        assert sorted(resultats) == list(range(1, 21)), "appels sérialisés : aucun n'est perdu ni dupliqué"

    def test_avec_des_fonctions_d_hote_on_garde_le_pont_par_processus(self, tmp_path: Path) -> None:
        hote = _Hote()
        with SubprocessSandbox(timeout=10, warm=True) as bac:
            outil = DynamicToolBuilder(Sequence([_payload(CODE_HOTE)]), tools_dir=tmp_path, sandbox=bac,
                                       host_functions={"lire": hote.lire}).build(ToolBuildRequest(capability="x"))
            assert outil(cle="q") == {"v": {"valeur": "Q"}}
            assert bac._workers == {}, "le pont n'utilise pas de worker chaud"

    def test_beaucoup_plus_rapide_que_le_froid(self, tmp_path: Path) -> None:
        fichier = tmp_path / "compteur.py"
        fichier.write_text(COMPTEUR, encoding="utf-8")
        froid = load_generated_tool(fichier, sandbox=SubprocessSandbox(timeout=10))
        t0 = time.perf_counter()
        for _ in range(10):
            froid()
        t_froid = time.perf_counter() - t0
        with SubprocessSandbox(timeout=10, warm=True) as bac:
            chaud = load_generated_tool(fichier, sandbox=bac)
            t0 = time.perf_counter()
            for _ in range(10):
                chaud()
            t_chaud = time.perf_counter() - t0
        assert t_chaud < t_froid * 0.6, f"chaud {t_chaud:.2f}s contre froid {t_froid:.2f}s"
