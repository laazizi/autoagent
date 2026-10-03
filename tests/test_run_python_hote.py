"""`run_python` + fonctions de l'hôte (0.22.0) : un programme au lieu d'un appel d'outil par tour.

Le modèle écrit UN extrait qui appelle plusieurs fois des fonctions de l'hôte
(`context["call_host"]`), au lieu d'émettre un appel d'outil par tour. Opt-in :
sans `host_functions`, `run_python` est exactement ce qu'il était.
"""

from __future__ import annotations

import pytest

from autoagent import Agent, PythonRunner
from autoagent.errors import ToolError
from autoagent.providers.base import LLMProvider
from autoagent.schema import LLMRequest, LLMResponse, ModelConfig, ToolCall


def lire(capteur: str, jour: str) -> dict:
    """Comptage d'un capteur pour un jour."""
    return {"capteur": capteur, "n": len(capteur) * 10 + int(jour[-2:])}


BOUCLE = (
    "def run(args, context):\n"
    "    total = 0\n"
    "    for c in args['capteurs']:\n"
    "        total += context['call_host']('lire', {'capteur': c, 'jour': '2026-09-01'})['n']\n"
    "    return {'total': total}\n"
)


class TestPythonRunnerAvecHote:
    def test_un_extrait_appelle_plusieurs_fois_l_hote(self) -> None:
        runner = PythonRunner(host_functions={"lire": lire})
        sortie = runner(BOUCLE, {"capteurs": ["a", "bb", "ccc"]})
        assert sortie["result"] == {"total": (10 + 1) + (20 + 1) + (30 + 1)}

    def test_sans_fonctions_rien_ne_change(self) -> None:
        runner = PythonRunner()
        assert runner.host_functions is None and runner.describe_host_functions() == ""
        with pytest.raises(ToolError, match="call_host"):
            runner(BOUCLE, {"capteurs": ["a"]})

    def test_un_nom_hors_liste_blanche_est_refuse(self) -> None:
        code = "def run(args, context):\n    return context['call_host']('supprimer', {})\n"
        with pytest.raises(ToolError, match="host function not allowed: supprimer"):
            PythonRunner(host_functions={"lire": lire})(code)

    def test_une_exception_de_l_hote_est_rendue_au_modele(self) -> None:
        def casse(capteur: str) -> dict:
            raise KeyError(capteur)

        code = "def run(args, context):\n    return context['call_host']('casse', {'capteur': 'zz'})\n"
        with pytest.raises(ToolError, match="KeyError"):
            PythonRunner(host_functions={"casse": casse})(code)

    def test_la_description_donne_noms_signatures_et_docstring(self) -> None:
        texte = PythonRunner(host_functions={"lire": lire}).describe_host_functions()
        assert "context['call_host'](name" in texte
        assert "- lire(capteur: 'str', jour: 'str') -> 'dict': Comptage d'un capteur pour un jour." in texte


class _Principal(LLMProvider):
    def __init__(self, reponses: list[LLMResponse]) -> None:
        super().__init__(ModelConfig(provider="p", model="p", api_key="x"))
        self.reponses = list(reponses)
        self.requetes: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requetes.append(request)
        return self.reponses.pop(0)


class TestDansLAgent:
    def test_la_description_de_l_outil_liste_les_fonctions_de_l_hote(self) -> None:
        agent = Agent(_Principal([LLMResponse(content="ok")]))
        agent.enable_run_python(PythonRunner(host_functions={"lire": lire}))
        spec = next(s for s in agent.registry.specs() if s.name == "run_python")
        assert "- lire(capteur: 'str', jour: 'str')" in spec.description

    def test_sans_fonctions_la_description_est_celle_d_avant(self) -> None:
        agent = Agent(_Principal([LLMResponse(content="ok")]))
        agent.enable_run_python()
        spec = next(s for s in agent.registry.specs() if s.name == "run_python")
        assert "call_host" not in spec.description

    def test_un_run_complet_trente_six_lectures_en_un_seul_appel_d_outil(self) -> None:
        capteurs = [f"c{i:02d}" for i in range(12)]
        appel = ToolCall(id="c1", name="run_python", arguments={"code": BOUCLE, "args": {"capteurs": capteurs}})
        principal = _Principal([LLMResponse(tool_calls=[appel]), LLMResponse(content="fini")])
        agent = Agent(principal, max_steps=4)
        agent.enable_run_python(PythonRunner(host_functions={"lire": lire}))
        resultat = agent.run("go")
        attendu = sum(len(c) * 10 + 1 for c in capteurs)
        outil = next(m for m in resultat.messages if m.role == "tool")
        assert str(attendu) in outil.content
        assert resultat.steps == 2, "un seul appel d'outil pour douze lectures"
