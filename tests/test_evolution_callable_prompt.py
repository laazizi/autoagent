"""`enable_software_evolution` avec un `system_prompt` CALLABLE (0.21.0).

Le seul des 14 signalements mypy « antérieurs » qui était un VRAI bug : un
agent à prompt dynamique (`system_prompt=lambda: ...`) ne pouvait pas activer
l'évolution — `TypeError: argument of type 'function' is not iterable`.
Reproduit, corrigé, verrouillé ici.
"""

from __future__ import annotations

from pathlib import Path

from autoagent import Agent
from autoagent.evolution import EVOLUTION_SYSTEM_PROMPT, EvolutionRuntime, enable_software_evolution

from .conftest import FakeLLMProvider

MARQUEUR = EVOLUTION_SYSTEM_PROMPT.strip()


def _runtime(tmp_path: Path) -> EvolutionRuntime:
    return EvolutionRuntime(tmp_path)


class TestPromptCallable:
    def test_ne_plante_plus_et_ajoute_les_consignes(self, tmp_path: Path) -> None:
        agent = Agent(FakeLLMProvider([]), system_prompt=lambda: "prompt dynamique")
        enable_software_evolution(agent, _runtime(tmp_path))
        assert callable(agent.system_prompt), "le prompt doit RESTER dynamique"
        texte = agent.system_prompt()
        assert texte.startswith("prompt dynamique") and MARQUEUR in texte

    def test_reste_dynamique(self, tmp_path: Path) -> None:
        etat = {"n": 1}
        agent = Agent(FakeLLMProvider([]), system_prompt=lambda: f"tour {etat['n']}")
        enable_software_evolution(agent, _runtime(tmp_path))
        assert agent.system_prompt().startswith("tour 1")
        etat["n"] = 2
        assert agent.system_prompt().startswith("tour 2"), "l'enveloppe a figé le prompt"

    def test_idempotent_pas_de_double_ajout(self, tmp_path: Path) -> None:
        agent = Agent(FakeLLMProvider([]), system_prompt=lambda: "base")
        enable_software_evolution(agent, _runtime(tmp_path))
        enable_software_evolution(agent, _runtime(tmp_path))
        assert agent.system_prompt().count(MARQUEUR) == 1


class TestPromptChaine:
    def test_comportement_historique_inchange(self, tmp_path: Path) -> None:
        agent = Agent(FakeLLMProvider([]), system_prompt="prompt statique")
        enable_software_evolution(agent, _runtime(tmp_path))
        assert isinstance(agent.system_prompt, str)
        assert agent.system_prompt.startswith("prompt statique") and MARQUEUR in agent.system_prompt
        enable_software_evolution(agent, _runtime(tmp_path))
        assert agent.system_prompt.count(MARQUEUR) == 1
