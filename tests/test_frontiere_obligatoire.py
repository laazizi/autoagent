"""`Agent(require_boundary=True)` : le code écrit par le modèle ne tourne QUE derrière une vraie frontière.

Avant : exiger une frontière se disait bac par bac (`make_sandbox(require_boundary=True)`), à trois endroits
(`enable_dynamic_tools`, `enable_run_python`, `enable_code_action`). Sans bac choisi, la bibliothèque prenait le
confinement par le noyau là où il existe, sinon le sous-processus SIMPLE — une liste d'interdits, pas une
frontière — même quand un démon Docker répondait (constaté le 12 nov. 2026 sur une machine Windows : Docker
détecté, bac par défaut = sous-processus simple).

Le contrat fixé ici :

* un seul réglage, sur l'agent, défaut `False` (rien ne change pour un agent existant) ;
* à la CONFIGURATION : un bac par défaut est remplacé par le confinement, sinon Docker, sinon refus ; un bac choisi
  par l'hôte sans frontière est refusé tout de suite ;
* à l'EXÉCUTION, en plus (fail-closed) : un bac remplacé après coup, un outil généré chargé à la main, le run d'un
  sous-agent délégué par un agent qui exige la frontière — rien ne tourne sans frontière ;
* l'évolution logicielle (qui fait tourner la commande de validation de l'hôte sur du code édité par le modèle,
  hors de tout bac) est refusée sur un tel agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from autoagent import Agent, DynamicToolBuilder, PythonRunner, delegate_to
from autoagent import sandbox as module_bac
from autoagent.errors import ToolError
from autoagent.providers.base import LLMProvider
from autoagent.sandbox import DockerSandbox, GeneratedPythonTool, SubprocessSandbox
from autoagent.schema import LLMRequest, LLMResponse, ModelConfig, ToolCall, ToolSpec


class _Scripte(LLMProvider):
    def __init__(self, reponses: list[LLMResponse]) -> None:
        super().__init__(ModelConfig(provider="p", model="p", api_key="x"))
        self.reponses = list(reponses)

    def complete(self, request: LLMRequest) -> LLMResponse:
        return self.reponses.pop(0) if self.reponses else LLMResponse(content="fini")


class _BacAvecFrontiere:
    """Un bac qui se déclare frontière de l'OS ; il n'exécute rien, il compte."""

    timeout = 10.0
    frontiere = True

    def __init__(self) -> None:
        self.appels = 0

    def isolation(self) -> dict[str, Any]:
        return {"kind": "faux", "os_boundary": self.frontiere}

    def run_python_tool(self, file_path: Any, args: Any, context: Any = None, *,
                        allow_network: bool = False, host_functions: Any = None) -> dict[str, Any]:
        self.appels += 1
        return {"ok": True, "result": "dans le bac", "stdout": ""}


class _BacSansFrontiere(_BacAvecFrontiere):
    frontiere = False


PROGRAMME = "def run(args, context):\n    return 42\n"


def _programme() -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id="c-prog", name="run_python", arguments={"code": PROGRAMME})])


def _appel(nom: str, **arguments: Any) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id=f"c-{nom}", name=nom, arguments=arguments)])


def _resultats_outils(resultat: Any) -> str:
    return "\n".join(m.content or "" for m in resultat.messages if m.role == "tool")


@pytest.fixture
def ni_noyau_ni_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module_bac, "_confinement_impossible", lambda: "noyau sans Landlock (test)")
    monkeypatch.setattr(module_bac, "docker_available", lambda: False)


@pytest.fixture
def docker_seulement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module_bac, "_confinement_impossible", lambda: "noyau sans Landlock (test)")
    monkeypatch.setattr(module_bac, "docker_available", lambda: True)


class TestLeReglage:
    def test_defaut_faux(self) -> None:
        assert Agent(_Scripte([])).require_boundary is False

    def test_un_booleen_seulement(self) -> None:
        with pytest.raises(TypeError, match="require_boundary"):
            Agent(_Scripte([]), require_boundary="oui")  # type: ignore[arg-type]


class TestALaConfiguration:
    def test_sans_le_reglage_rien_ne_change(self, docker_seulement: None) -> None:
        """Non-régression : Docker disponible, mais le bac par défaut reste celui d'avant."""
        agent = Agent(_Scripte([]))
        agent.enable_code_action()
        assert isinstance(agent.python_runner.sandbox, SubprocessSandbox)
        assert agent.python_runner.sandbox.confine is None

    def test_le_bac_par_defaut_devient_docker(self, docker_seulement: None) -> None:
        agent = Agent(_Scripte([]), require_boundary=True)
        agent.enable_code_action()
        assert isinstance(agent.python_runner.sandbox, DockerSandbox)

    def test_ni_noyau_ni_docker_refus_tout_de_suite(self, ni_noyau_ni_docker: None) -> None:
        agent = Agent(_Scripte([]), require_boundary=True)
        with pytest.raises(ToolError, match="noyau sans Landlock") as refus:
            agent.enable_code_action()
        assert "require_boundary" in str(refus.value)
        assert "run_python" not in agent.registry               # rien n'est à moitié configuré

    def test_un_bac_choisi_sans_frontiere_est_refuse(self, docker_seulement: None) -> None:
        """Le choix de l'hôte n'est jamais remplacé en silence : il est refusé, avec la raison."""
        agent = Agent(_Scripte([]), require_boundary=True)
        with pytest.raises(ToolError, match="require_boundary"):
            agent.enable_code_action(sandbox=_BacSansFrontiere())

    def test_un_bac_choisi_avec_frontiere_est_garde(self, ni_noyau_ni_docker: None) -> None:
        bac = _BacAvecFrontiere()
        agent = Agent(_Scripte([]), require_boundary=True)
        agent.enable_code_action(sandbox=bac)
        assert agent.python_runner.sandbox is bac

    def test_enable_run_python(self, docker_seulement: None) -> None:
        agent = Agent(_Scripte([]), require_boundary=True)
        agent.enable_run_python(PythonRunner())
        assert isinstance(agent.python_runner.sandbox, DockerSandbox)
        autre = Agent(_Scripte([]), require_boundary=True)
        with pytest.raises(ToolError, match="require_boundary"):
            autre.enable_run_python(PythonRunner(_BacSansFrontiere()))  # type: ignore[arg-type]

    def test_enable_dynamic_tools(self, docker_seulement: None, tmp_path: Path) -> None:
        agent = Agent(_Scripte([]), require_boundary=True)
        constructeur = DynamicToolBuilder(_Scripte([]), tools_dir=tmp_path / "a")
        agent.enable_dynamic_tools(constructeur)
        assert isinstance(constructeur.sandbox, DockerSandbox)
        # Un outil qui porte une permission de fichiers ne retombe PAS dans le sous-processus simple : il garde
        # le conteneur du constructeur (et ses réglages).
        assert constructeur._bac_pour(["filesystem.read"]) is constructeur.sandbox
        autre = Agent(_Scripte([]), require_boundary=True)
        impose = DynamicToolBuilder(_Scripte([]), tools_dir=tmp_path / "b", sandbox=_BacSansFrontiere())  # type: ignore[arg-type]
        with pytest.raises(ToolError, match="require_boundary"):
            autre.enable_dynamic_tools(impose)

    def test_outil_a_fichiers_sans_docker_refuse(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Bac confiné par le noyau, mais un outil à permission de fichiers n'a pas de dossiers nommés : sans
        Docker, il est refusé au lieu de tourner dans le sous-processus simple."""
        monkeypatch.setattr(module_bac, "docker_available", lambda: False)
        constructeur = DynamicToolBuilder(_Scripte([]), tools_dir=tmp_path, sandbox=None)
        constructeur.sandbox = _BacAvecFrontiere()  # type: ignore[assignment]   # le confinement, tel qu'un Linux le rendrait
        agent = Agent(_Scripte([]), require_boundary=True)
        agent.enable_dynamic_tools(constructeur)
        assert constructeur._bac_pour([]) is constructeur.sandbox
        with pytest.raises(ToolError, match="filesystem"):
            constructeur._bac_pour(["filesystem.write"])

    def test_le_noyau_ne_confine_pas_des_fichiers_non_nommes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Même là où le confinement est possible, un outil à permission de fichiers sans dossiers nommés n'y
        va pas : une liste blanche ne s'invente pas. Docker, ou refus."""
        monkeypatch.setattr(module_bac, "_confinement_impossible", lambda: None)
        monkeypatch.setattr(module_bac, "docker_available", lambda: False)
        with pytest.raises(ToolError, match="filesystem"):
            module_bac._bac_avec_frontiere(["filesystem.read"])
        monkeypatch.setattr(module_bac, "docker_available", lambda: True)
        assert isinstance(module_bac._bac_avec_frontiere(["filesystem.read"]), DockerSandbox)

    def test_l_evolution_logicielle_est_refusee(self) -> None:
        agent = Agent(_Scripte([]), require_boundary=True)
        with pytest.raises(ToolError, match="require_boundary"):
            agent.enable_evolution(object())


class TestALExecution:
    def test_un_bac_avec_frontiere_execute(self) -> None:
        bac = _BacAvecFrontiere()
        agent = Agent(_Scripte([_programme()]), max_steps=4, require_boundary=True)
        agent.enable_code_action(sandbox=bac)
        assert "dans le bac" in _resultats_outils(agent.run("go"))
        assert bac.appels == 1

    def test_un_bac_remplace_apres_coup_ne_tourne_pas(self) -> None:
        agent = Agent(_Scripte([_programme()]), max_steps=4, require_boundary=True)
        agent.enable_code_action(sandbox=_BacAvecFrontiere())
        mauvais = _BacSansFrontiere()
        agent.python_runner.sandbox = mauvais  # type: ignore[assignment]
        assert "require_boundary" in _resultats_outils(agent.run("go"))
        assert mauvais.appels == 0

    def test_un_bac_qui_ne_sait_pas_le_dire_compte_pour_non(self) -> None:
        class Muet(_BacAvecFrontiere):
            def isolation(self) -> dict[str, Any]:
                raise RuntimeError("inconnu")

        agent = Agent(_Scripte([_programme()]), max_steps=4, require_boundary=True)
        agent.enable_code_action(sandbox=_BacAvecFrontiere())
        muet = Muet()
        agent.python_runner.sandbox = muet  # type: ignore[assignment]
        assert "require_boundary" in _resultats_outils(agent.run("go"))
        assert muet.appels == 0

    def _outil_genere(self, tmp_path: Path, bac: Any) -> GeneratedPythonTool:
        fichier = tmp_path / "g.py"
        fichier.write_text(PROGRAMME, encoding="utf-8")
        spec = ToolSpec(name="g", description="outil généré", input_schema={"type": "object", "properties": {}})
        return GeneratedPythonTool(spec=spec, file_path=fichier, sandbox=bac)

    def _hote_qui_appelle(self, outil: GeneratedPythonTool, **kw: Any) -> Agent:
        agent = Agent(_Scripte([_appel("appeler")]), max_steps=4, **kw)

        @agent.tool
        def appeler() -> str:
            """Appelle l'outil généré."""
            return str(outil())

        return agent

    def test_un_outil_genere_sans_frontiere_ne_tourne_pas(self, tmp_path: Path) -> None:
        bac = _BacSansFrontiere()
        agent = self._hote_qui_appelle(self._outil_genere(tmp_path, bac), require_boundary=True)
        assert "require_boundary" in _resultats_outils(agent.run("go"))
        assert bac.appels == 0

    def test_sans_le_reglage_l_outil_genere_tourne(self, tmp_path: Path) -> None:
        bac = _BacSansFrontiere()
        agent = self._hote_qui_appelle(self._outil_genere(tmp_path, bac))
        assert "dans le bac" in _resultats_outils(agent.run("go"))
        assert bac.appels == 1

    def test_hors_de_tout_run_rien_n_est_consulte(self, tmp_path: Path) -> None:
        """Comme la porte : l'hôte qui appelle un outil à la main décide seul."""
        bac = _BacSansFrontiere()
        assert self._outil_genere(tmp_path, bac)() == "dans le bac"
        assert bac.appels == 1


class TestLaFrontiereDescendDansLesSousAgents:
    """Comme la teinte (0.25.0) : un agent qui exige la frontière ne la perd pas en déléguant."""

    def _specialiste(self) -> tuple[Agent, _BacSansFrontiere]:
        bac = _BacSansFrontiere()
        specialiste = Agent(_Scripte([_programme(), LLMResponse(content="fait")]), max_steps=4)
        specialiste.enable_code_action(sandbox=bac)        # le spécialiste, lui, n'exige rien
        return specialiste, bac

    def test_as_tool(self) -> None:
        specialiste, bac = self._specialiste()
        parent = Agent(_Scripte([_appel("expert", request="calcule")]), max_steps=4, require_boundary=True)
        parent.add_tool(specialiste.as_tool(name="expert", description="Un expert."))
        parent.run("go")
        assert bac.appels == 0
        assert specialiste._frontiere_heritee is False      # l'héritage ne dure que la délégation

    def test_delegate_to(self) -> None:
        specialiste, bac = self._specialiste()
        parent = Agent(_Scripte([_appel("delegate", requests=[{"specialist": "expert", "request": "calcule"}])]),
                       max_steps=4, require_boundary=True)
        parent.add_tool(delegate_to({"expert": specialiste}))
        parent.run("go")
        assert bac.appels == 0
        assert specialiste._frontiere_heritee is False

    def test_seul_le_specialiste_s_execute(self) -> None:
        specialiste, bac = self._specialiste()
        specialiste.run("calcule")
        assert bac.appels == 1

    def test_un_parent_qui_n_exige_rien_ne_change_rien(self) -> None:
        specialiste, bac = self._specialiste()
        parent = Agent(_Scripte([_appel("expert", request="calcule")]), max_steps=4)
        parent.add_tool(specialiste.as_tool(name="expert", description="Un expert."))
        parent.run("go")
        assert bac.appels == 1


@pytest.mark.skipif(module_bac._confinement_impossible() is not None and not module_bac.docker_available(),
                    reason="ni confinement par le noyau ni Docker sur cette machine")
def test_reel_un_programme_tourne_derriere_une_vraie_frontiere() -> None:
    """Sans faux : la meilleure frontière de la machine (le noyau sous Linux, sinon Docker) exécute le code."""
    agent = Agent(_Scripte([_programme()]), max_steps=4, require_boundary=True)
    agent.enable_code_action(timeout=120.0)
    assert agent.python_runner.sandbox.isolation()["os_boundary"] is True
    assert '"result": 42' in _resultats_outils(agent.run("go"))
