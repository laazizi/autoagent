"""`run_k` — mesure de fiabilité `pass^k` (0.18.0).

`pass@1` ne dit presque rien à un exploitant : 90 % de pass@1 donne **57 % à
k=8**, parce que `pass^k ≈ p^k` s'effondre exponentiellement. Ce banc rend cet
effondrement visible, avec un juge DÉTERMINISTE fourni par l'hôte (jamais un
LLM-as-judge : ils plafonnent sous 55 % sur les échecs d'agent).
"""

from __future__ import annotations

import pytest

from autoagent import Agent
from autoagent.eval import run_k
from autoagent.schema import LLMResponse, ToolCall

from .conftest import FakeLLMProvider


def _agent(reponses) -> Agent:
    return Agent(FakeLLMProvider(list(reponses)), max_steps=4)


class TestMesures:
    def test_tout_reussi(self) -> None:
        rapport = run_k(lambda: _agent(["42"]), "combien ?", k=5,
                        check=lambda res: "42" in res.output)
        assert rapport.successes == 5
        assert rapport.pass_at_1 == 1.0
        assert rapport.pass_hat_k == 1.0
        assert rapport.estimated_pass_hat_k == 1.0

    def test_tout_echoue(self) -> None:
        rapport = run_k(lambda: _agent(["je ne sais pas"]), "combien ?", k=4,
                        check=lambda res: "42" in res.output)
        assert rapport.successes == 0
        assert rapport.pass_at_1 == 0.0
        assert rapport.pass_hat_k == 0.0

    def test_l_effondrement_est_visible(self) -> None:
        """LE point du module : 75 % de pass@1 → 32 % de pass^4 estimé."""
        sequences = [["42"], ["42"], ["42"], ["raté"]]
        it = iter(sequences)
        rapport = run_k(lambda: _agent(next(it)), "combien ?", k=4,
                        check=lambda res: "42" in res.output)
        assert rapport.pass_at_1 == 0.75
        assert rapport.pass_hat_k == 0.0                      # observé : pas toutes
        assert round(rapport.estimated_pass_hat_k, 4) == 0.3164   # 0.75^4

    def test_dispersion_des_etapes(self) -> None:
        """Un agent régulier et un agent erratique n'ont pas la même fiabilité,
        même à pass@1 égal — la dispersion doit être visible."""
        sequences = [
            ["42"],                                                   # 1 étape
            [LLMResponse(tool_calls=[ToolCall(id="c", name="outil", arguments={})]),
             "42"],                                                   # 2 étapes
        ]
        it = iter(sequences)

        def fabrique() -> Agent:
            a = _agent(next(it))

            @a.tool
            def outil() -> dict:
                """Un outil."""
                return {"ok": True}

            return a

        rapport = run_k(fabrique, "combien ?", k=2, check=lambda res: "42" in res.output)
        assert rapport.steps_range == (1, 2)
        assert rapport.median_steps == 1.5


class TestRobustesse:
    def test_un_run_qui_plante_est_un_echec_pas_une_erreur_de_banc(self) -> None:
        class ProviderQuiPlante:
            def __init__(self) -> None:
                from autoagent.schema import ModelConfig
                self.config = ModelConfig(provider="fake", model="fake")

            def complete(self, request):
                raise RuntimeError("502 du provider")

        rapport = run_k(lambda: Agent(ProviderQuiPlante()), "vas-y", k=3,
                        check=lambda res: True)
        assert rapport.successes == 0
        assert len(rapport.errors) == 3
        assert "502" in rapport.errors[0]

    def test_un_juge_casse_ne_passe_pas_inapercu(self) -> None:
        """Un `check` qui lève ne doit pas être avalé en silence."""
        def juge_casse(res):
            raise KeyError("clé absente")

        rapport = run_k(lambda: _agent(["42"]), "combien ?", k=2, check=juge_casse)
        assert rapport.successes == 0
        assert "check raised" in rapport.errors[0]

    def test_agent_unique_accepte(self) -> None:
        """Un Agent (pas une fabrique) est accepté — pratique pour un agent
        sans effet de bord."""
        agent = _agent(["42", "42"])
        rapport = run_k(agent, "combien ?", k=2, check=lambda res: "42" in res.output)
        assert rapport.successes == 2

    def test_rappel_de_progression(self) -> None:
        vus: list[int] = []
        run_k(lambda: _agent(["42"]), "combien ?", k=3,
              check=lambda res: True, on_attempt=lambda a: vus.append(a.index))
        assert vus == [1, 2, 3]

    def test_rappel_qui_plante_est_fail_open(self) -> None:
        """L'observabilité ne casse jamais la mesure (même contrat que la trace)."""
        def casse(attempt):
            raise RuntimeError("boom")

        rapport = run_k(lambda: _agent(["42"]), "combien ?", k=2,
                        check=lambda res: True, on_attempt=casse)
        assert rapport.successes == 2


class TestGardes:
    def test_k_invalide(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            run_k(lambda: _agent(["x"]), "t", k=0, check=lambda r: True)

    def test_check_non_callable(self) -> None:
        with pytest.raises(TypeError, match="check must be"):
            run_k(lambda: _agent(["x"]), "t", k=1, check="pas un callable")


class TestExport:
    def test_to_dict_est_json_safe(self) -> None:
        import json
        rapport = run_k(lambda: _agent(["42"]), "combien ?", k=2,
                        check=lambda res: "42" in res.output)
        json.dumps(rapport.to_dict())          # ne doit pas lever
        assert rapport.to_dict()["successes"] == 2

    def test_summary_lisible(self) -> None:
        rapport = run_k(lambda: _agent(["42"]), "combien ?", k=2,
                        check=lambda res: "42" in res.output)
        assert "pass@1=1.00" in rapport.summary()
        assert "pass^2" in rapport.summary()
