"""`Bounds` — les huit bornes d'un agent en un objet (0.21.0).

Ce que ces tests verrouillent :

  1. `Bounds()` a EXACTEMENT les défauts d'`Agent(...)` : passer `Bounds()` ne
     change rien.
  2. Les huit champs se posent d'un coup ; un kwarg EXPLICITE l'emporte sur le
     champ correspondant (l'intention la plus locale gagne).
  3. `agent.bounds` est une PHOTO de ce qui est en vigueur : elle suit les
     attributs modifiés après construction (la démo 22 relève `token_budget`
     avant un `resume`).
  4. Rétrocompatibilité : les vingt kwargs marchent comme avant.
"""

from __future__ import annotations

from autoagent import Agent, Bounds

from .conftest import FakeLLMProvider


def _agent(**kw) -> Agent:  # type: ignore[no-untyped-def]
    return Agent(FakeLLMProvider([]), **kw)


class TestDefauts:
    def test_bounds_vide_egale_les_defauts_d_agent(self) -> None:
        sans, avec = _agent(), _agent(bounds=Bounds())
        for nom in Bounds.names():
            assert getattr(sans, nom) == getattr(avec, nom), nom

    def test_la_photo_d_un_agent_par_defaut_est_bounds_par_defaut(self) -> None:
        assert _agent().bounds == Bounds()


class TestApplication:
    def test_les_huit_bornes_d_un_coup(self) -> None:
        b = Bounds(max_steps=12, token_budget=40_000, max_tool_result_chars=4_000,
                   prune_tool_results_after=2, prune_batch=3, max_repeated_tool_calls=3,
                   trifecta_guard="approve", shadow_guards=True)
        a = _agent(bounds=b)
        assert a.bounds == b
        assert (a.max_steps, a.token_budget, a.prune_batch, a.trifecta_guard) == (12, 40_000, 3, "approve")

    def test_un_kwarg_explicite_l_emporte(self) -> None:
        a = _agent(bounds=Bounds(max_steps=12, token_budget=40_000), token_budget=500)
        assert a.max_steps == 12, "le champ non contredit vient de Bounds"
        assert a.token_budget == 500, "le kwarg explicite gagne"

    def test_un_kwarg_laisse_a_sa_valeur_par_defaut_ne_contredit_pas(self) -> None:
        # max_steps=8 est la valeur par défaut : on ne peut pas savoir si l'appelant
        # l'a écrite ; Bounds l'emporte alors — documenté, et c'est le comportement
        # le moins surprenant pour qui passe un objet de bornes.
        a = _agent(bounds=Bounds(max_steps=12), max_steps=8)
        assert a.max_steps == 12

    def test_prune_batch_borne_a_1(self) -> None:
        assert _agent(bounds=Bounds(prune_batch=0)).prune_batch == 1


class TestPhoto:
    def test_bounds_suit_les_attributs_modifies(self) -> None:
        a = _agent(token_budget=600)
        assert a.bounds.token_budget == 600
        a.token_budget = 8000                     # ce que fait la démo 22 avant resume
        assert a.bounds.token_budget == 8000

    def test_to_dict_serialisable(self) -> None:
        d = _agent(bounds=Bounds(max_steps=3)).bounds.to_dict()
        assert d["max_steps"] == 3 and set(d) == set(Bounds.names())

    def test_bounds_est_immuable(self) -> None:
        b = Bounds()
        try:
            b.max_steps = 99  # type: ignore[misc]
        except AttributeError:
            return
        raise AssertionError("Bounds doit être frozen")
