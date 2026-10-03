"""`compare_configs` — comparer deux configurations sans se raconter d'histoires.

Ce fichier teste le COMPORTEMENT : les calculs sur des cas qu'on peut vérifier à la
main ou par force brute, puis le banc (alternance des bras, fabrique neuve à chaque
tentative, plan trop petit, contrôle A/A, empreintes, coût). Les garanties
statistiques — taux de fausses alertes, couverture, puissance — sont dans
`test_compare_calibration.py`, mesurées par énumération et par simulation.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from typing import Any

import pytest

from autoagent import Agent, EvalTask, Variant, compare_configs
from autoagent import compare as cmp
from autoagent.compare import (
    A_BETTER,
    B_BETTER,
    INDISTINGUISHABLE,
    detectable_difference,
    paired_interval,
    paired_p_value,
    wilson_interval,
)
from autoagent.schema import TokenUsage

from .conftest import FakeLLMProvider

# ── Des agents factices dont on contrôle exactement les réussites ────────────


class _Res:
    def __init__(self, ok: bool, tokens: int | None) -> None:
        self.output = "ok" if ok else "ko"
        self.steps = 1
        self.usage = (None if tokens is None else
                      TokenUsage(input_tokens=tokens - 10, output_tokens=10, total_tokens=tokens))


def _bras(journal: list[tuple[str, str]], nom: str, succes: dict[str, int],
          jetons: int | None = 100):
    """Fabrique d'agents factices : `succes[prompt]` réussites sur cette consigne, dans
    l'ordre des appels (indépendant de l'ordre d'alternance des bras)."""
    compteurs: Counter[str] = Counter()
    fabriques: list[int] = []

    class _Agent:
        def run(self, prompt: str, context: dict[str, Any] | None = None) -> _Res:
            journal.append((nom, prompt))
            rang = compteurs[prompt]
            compteurs[prompt] += 1
            return _Res(rang < succes[prompt], jetons)

    def fabrique() -> _Agent:
        fabriques.append(1)
        return _Agent()

    fabrique.appels = fabriques  # type: ignore[attr-defined]
    return fabrique


def _bras_p(p: float, graine: int, jetons: int | None = 100):
    """Fabrique d'agents factices SANS état partagé de comptage : chaque tentative réussit
    avec la probabilité `p`. Pour les tests de contrôle A/A, où le bras A et sa copie
    appellent la MÊME fabrique : un compteur partagé les coupleraient."""
    import random

    rng = random.Random(graine)

    class _Agent:
        def run(self, prompt: str, context: dict[str, Any] | None = None) -> _Res:
            return _Res(rng.random() < p, jetons)

    return lambda: _Agent()


def _juge(res: Any) -> bool:
    return bool(res.output == "ok")


def _taches(n: int = 4) -> list[EvalTask]:
    return [EvalTask(f"t{i}", f"t{i}", _juge) for i in range(1, n + 1)]


def _succes(valeur: int, n: int = 4) -> dict[str, int]:
    return {f"t{i}": valeur for i in range(1, n + 1)}


# ── Calculs ──────────────────────────────────────────────────────────────────


class TestIntervalle:
    def test_wilson_aux_bords(self) -> None:
        """5/5 ne veut pas dire « 100 % ± 0 » : le bas reste loin de 1, le haut vaut 1."""
        z2 = 1.959964 ** 2
        bas0, haut0 = wilson_interval(0, 5)
        assert bas0 == 0.0
        assert haut0 == pytest.approx(z2 / (5 + z2), abs=1e-4)          # forme fermée de 0/n
        bas5, haut5 = wilson_interval(5, 5)
        assert haut5 == 1.0
        assert bas5 == pytest.approx(1 - z2 / (5 + z2), abs=1e-4)

    def test_wilson_est_symetrique(self) -> None:
        bas, haut = wilson_interval(2, 7)
        bas_s, haut_s = wilson_interval(5, 7)
        assert bas == pytest.approx(1 - haut_s)
        assert haut == pytest.approx(1 - bas_s)

    def test_exemple_de_newcombe(self) -> None:
        """56/70 contre 48/80 : l'exemple de Newcombe (Statistics in Medicine 17:873-890,
        1998), construction « score de Wilson ». Valeurs recoupées avec l'implémentation
        indépendante de statsmodels 0.15.0 (`confint_proportions_2indep`,
        `method="newcomb"`) : écart 0,2000, intervalle [0,0524 ; 0,3339]."""
        delta, bas, haut = paired_interval([(48, 80, 56, 70)])
        assert round(delta, 4) == 0.2
        assert round(bas, 4) == 0.0524
        assert round(haut, 4) == 0.3339

    def test_l_ecart_est_la_moyenne_des_ecarts_par_tache(self) -> None:
        delta, _bas, _haut = paired_interval([(1, 5, 4, 5), (3, 5, 3, 5), (0, 5, 5, 5)])
        assert delta == pytest.approx(((4 - 1) / 5 + 0 + (5 - 0) / 5) / 3)

    def test_plus_de_taches_resserre_l_intervalle(self) -> None:
        une = paired_interval([(1, 5, 4, 5)])
        quatre = paired_interval([(1, 5, 4, 5)] * 4)
        assert (quatre[2] - quatre[1]) < (une[2] - une[1])
        assert quatre[0] == pytest.approx(une[0])

    def test_validations(self) -> None:
        with pytest.raises(ValueError):
            paired_interval([])
        with pytest.raises(ValueError):
            paired_interval([(6, 5, 1, 5)])
        with pytest.raises(ValueError):
            wilson_interval(1, 0)
        with pytest.raises(ValueError):
            wilson_interval(1, 5, confidence=1.0)


def _p_force_brute(pairs: list[tuple[int, int, int, int]]) -> float:
    """p-value par ÉNUMÉRATION de toutes les permutations d'étiquettes, tâche par tâche :
    la définition du test, sans la convolution du module."""
    n = pairs[0][1]
    totaux = [sa + sb for sa, _na, sb, _nb in pairs]
    k = sum(totaux)
    observe = abs(k - 2 * sum(sa for sa, _na, _sb, _nb in pairs))
    comptes: Counter[int] = Counter()
    par_tache = []
    for kt in totaux:
        essais = [1] * kt + [0] * (2 * n - kt)
        par_tache.append(Counter(sum(essais[i] for i in idx)
                                 for idx in itertools.combinations(range(2 * n), n)))
    total = math.comb(2 * n, n) ** len(pairs)
    for combo in itertools.product(*[list(c.items()) for c in par_tache]):
        comptes[sum(x for x, _m in combo)] += math.prod(m for _x, m in combo)
    return sum(m for s, m in comptes.items() if abs(k - 2 * s) >= observe) / total


class TestTestExact:
    def test_valeurs_calculables_a_la_main(self) -> None:
        # n=1 : jamais significatif. Séparation parfaite : 2 / C(2n, n)^T.
        assert paired_p_value([(1, 1, 0, 1)]) == pytest.approx(1.0)
        assert paired_p_value([(2, 2, 0, 2)]) == pytest.approx(2 / 6)
        assert paired_p_value([(3, 3, 0, 3)]) == pytest.approx(2 / 20)
        assert paired_p_value([(4, 4, 0, 4)]) == pytest.approx(2 / 70)
        assert paired_p_value([(2, 2, 0, 2)] * 2) == pytest.approx(2 / 36)

    @pytest.mark.parametrize("pairs", [
        [(3, 3, 0, 3), (2, 3, 1, 3)],
        [(1, 3, 3, 3), (0, 3, 2, 3)],
        [(2, 3, 2, 3), (1, 3, 1, 3)],
        [(3, 3, 3, 3), (0, 3, 0, 3)],            # tâches sans information
        [(2, 4, 4, 4)],
        [(0, 3, 3, 3), (3, 3, 0, 3), (1, 3, 2, 3)],
    ])
    def test_egale_la_force_brute(self, pairs: list[tuple[int, int, int, int]]) -> None:
        assert paired_p_value(pairs) == pytest.approx(_p_force_brute(pairs), abs=1e-12)

    def test_symetrique_en_a_et_b_et_insensible_a_l_ordre_des_taches(self) -> None:
        pairs = [(1, 5, 4, 5), (3, 5, 5, 5), (0, 5, 2, 5)]
        inverse = [(sb, nb, sa, na) for sa, na, sb, nb in pairs]
        assert paired_p_value(pairs) == pytest.approx(paired_p_value(inverse))
        assert paired_p_value(pairs) == pytest.approx(paired_p_value(list(reversed(pairs))))

    def test_exige_le_meme_nombre_d_essais(self) -> None:
        with pytest.raises(ValueError, match="same number"):
            paired_p_value([(1, 5, 2, 5), (1, 4, 2, 4)])
        with pytest.raises(ValueError, match="same number"):
            paired_p_value([(1, 5, 2, 4)])


# ── Le banc ──────────────────────────────────────────────────────────────────


class TestVerdict:
    def test_un_ecart_net_est_detecte(self) -> None:
        j: list[tuple[str, str]] = []
        r = compare_configs(_bras(j, "A", _succes(2, 4)), _bras(j, "B", _succes(8, 4)),
                            _taches(), repeats=8)
        assert r.verdict == B_BETTER
        assert r.delta == pytest.approx(0.75)
        assert r.p_value < 0.05 and r.low > 0
        assert r.trustworthy and r.detectable is None
        assert "B MEILLEURE" in r.summary()

    def test_dans_l_autre_sens(self) -> None:
        j: list[tuple[str, str]] = []
        r = compare_configs(_bras(j, "A", _succes(8, 4)), _bras(j, "B", _succes(2, 4)),
                            _taches(), repeats=8)
        assert r.verdict == A_BETTER
        assert r.delta == pytest.approx(-0.75)

    def test_par_defaut_indistinguable_et_dit_ce_que_la_mesure_voit(self) -> None:
        j: list[tuple[str, str]] = []
        r = compare_configs(_bras(j, "A", _succes(5, 4)), _bras(j, "B", _succes(6, 4)),
                            _taches(), repeats=8)
        assert r.verdict == INDISTINGUISHABLE
        assert r.detectable is not None and r.detectable >= 0.25
        assert "PAS « équivalentes »" in r.summary()

    def test_les_noms_par_defaut(self) -> None:
        j: list[tuple[str, str]] = []
        r = compare_configs(_bras(j, "x", _succes(1)), _bras(j, "y", _succes(1)),
                            _taches(), repeats=2)
        assert (r.a, r.b) == ("A", "B")


class TestBanc:
    def test_les_bras_alternent(self) -> None:
        """À chaque répétition les deux bras passent, et celui qui passe en premier change
        d'une répétition à l'autre : une dérive ou un cache touche les deux."""
        j: list[tuple[str, str]] = []
        compare_configs(_bras(j, "A", _succes(3)), _bras(j, "B", _succes(3)), _taches(3),
                        repeats=6)
        for t in ("t1", "t2", "t3"):
            suite = [bras for bras, prompt in j if prompt == t]
            assert len(suite) == 12
            blocs = [suite[i:i + 2] for i in range(0, 12, 2)]
            assert all(sorted(b) == ["A", "B"] for b in blocs)
            premiers = [b[0] for b in blocs]
            assert all(premiers[i] != premiers[i + 1] for i in range(5))   # alternance stricte
            assert premiers.count("A") == premiers.count("B") == 3         # équilibré

    def test_trois_bras_tournent_aussi(self) -> None:
        j: list[tuple[str, str]] = []
        compare_configs(_bras(j, "A", _succes(3)), _bras(j, "B", _succes(3)), _taches(1),
                        repeats=6, control=True)
        # Le contrôle réutilise la fabrique de A : on le repère par l'ordre d'appel.
        assert len(j) == 18

    def test_la_graine_fixe_l_ordre(self) -> None:
        def ordre(graine: int) -> list[str]:
            j: list[tuple[str, str]] = []
            compare_configs(_bras(j, "A", _succes(3)), _bras(j, "B", _succes(3)), _taches(4),
                            repeats=2, seed=graine)
            return [b for b, _p in j]

        assert ordre(7) == ordre(7)
        assert len({tuple(ordre(s)) for s in range(8)}) > 1          # la graine change vraiment l'ordre

    def test_un_agent_neuf_par_tentative(self) -> None:
        j: list[tuple[str, str]] = []
        fa, fb = _bras(j, "A", _succes(2)), _bras(j, "B", _succes(2))
        compare_configs(fa, fb, _taches(3), repeats=4, control=True)
        assert len(fa.appels) == 3 * 4 * 2          # A + son contrôle  # type: ignore[attr-defined]
        assert len(fb.appels) == 3 * 4              # type: ignore[attr-defined]

    def test_une_instance_reutilisee_est_signalee(self) -> None:
        j: list[tuple[str, str]] = []
        instance = _bras(j, "A", _succes(9))()           # un AGENT, pas une fabrique
        r = compare_configs(instance, _bras(j, "B", _succes(2)), _taches(2), repeats=2)
        assert any("instance réutilisée" in n for n in r.notes)

    def test_une_exception_est_un_echec_et_se_voit(self) -> None:
        class Plante:
            def run(self, prompt: str, context: Any = None) -> Any:
                raise RuntimeError("502 du fournisseur")

        j: list[tuple[str, str]] = []
        r = compare_configs(_bras(j, "A", _succes(3)), lambda: Plante(), _taches(2), repeats=3)
        assert r.rate_b == 0.0
        assert any("502 du fournisseur" in n and "exception" in n for n in r.notes)

    def test_un_juge_qui_leve_est_signale(self) -> None:
        def casse(res: Any) -> bool:
            raise KeyError("clé absente")

        j: list[tuple[str, str]] = []
        taches = [EvalTask("t1", "t1", casse)]
        r = compare_configs(_bras(j, "A", {"t1": 3}), _bras(j, "B", {"t1": 3}), taches, repeats=3)
        assert r.rate_a == 0.0
        assert any("check raised" in n for n in r.notes)

    def test_plan_trop_petit(self) -> None:
        """Une tâche, trois essais : même une séparation parfaite donne p = 0,1."""
        j: list[tuple[str, str]] = []
        r = compare_configs(_bras(j, "A", {"t1": 0}), _bras(j, "B", {"t1": 3}), _taches(1),
                            repeats=3)
        assert r.p_min == pytest.approx(0.1)
        assert r.verdict == INDISTINGUISHABLE
        assert any("Plan trop petit" in n for n in r.notes)

    def test_plafond_signale(self) -> None:
        j: list[tuple[str, str]] = []
        r = compare_configs(_bras(j, "A", _succes(5)), _bras(j, "B", _succes(5)), _taches(3),
                            repeats=5)
        assert r.verdict == INDISTINGUISHABLE
        assert any("Plafond ou plancher" in n for n in r.notes)

    def test_les_taches_sans_information_sont_comptees(self) -> None:
        j: list[tuple[str, str]] = []
        sa = {"t1": 5, "t2": 1, "t3": 0}
        sb = {"t1": 5, "t2": 4, "t3": 0}
        r = compare_configs(_bras(j, "A", sa), _bras(j, "B", sb), _taches(3), repeats=5)
        assert any("2 tâche(s) sur 3 ne portent aucune information" in n for n in r.notes)

    def test_par_tache(self) -> None:
        j: list[tuple[str, str]] = []
        sa = {"t1": 1, "t2": 3, "t3": 4}
        sb = {"t1": 4, "t2": 3, "t3": 2}
        r = compare_configs(_bras(j, "A", sa), _bras(j, "B", sb), _taches(3), repeats=5)
        assert [x.direction for x in r.tasks] == ["b", "tie", "a"]
        assert [x.successes_a for x in r.tasks] == [1, 3, 4]
        assert "B > A sur 1, B < A sur 1, égalité sur 1" in r.summary()

    def test_validations(self) -> None:
        j: list[tuple[str, str]] = []
        fa, fb = _bras(j, "A", _succes(1)), _bras(j, "B", _succes(1))
        with pytest.raises(ValueError, match="repeats"):
            compare_configs(fa, fb, _taches(), repeats=0)
        with pytest.raises(ValueError, match="at least one task"):
            compare_configs(fa, fb, [])
        with pytest.raises(ValueError, match="unique"):
            compare_configs(fa, fb, [EvalTask("x", "t1", _juge), EvalTask("x", "t2", _juge)])
        with pytest.raises(TypeError, match="callable"):
            compare_configs(fa, fb, [EvalTask("x", "t1", "pas un juge")])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="distinct names"):
            compare_configs(Variant("v", fa), Variant("v", fb), _taches())
        with pytest.raises(ValueError, match="confidence"):
            compare_configs(fa, fb, _taches(), confidence=1.5)

    def test_le_rappel_qui_plante_ne_casse_pas_la_mesure(self) -> None:
        def casse(bras: str, tache: str, tentative: Any) -> None:
            raise RuntimeError("boom")

        j: list[tuple[str, str]] = []
        r = compare_configs(_bras(j, "A", _succes(2)), _bras(j, "B", _succes(2)), _taches(2),
                            repeats=2, on_attempt=casse)
        assert r.rate_a == 1.0

    def test_le_rappel_voit_chaque_tentative(self) -> None:
        vus: list[tuple[str, str, int]] = []
        j: list[tuple[str, str]] = []
        compare_configs(Variant("v1", _bras(j, "A", _succes(2))),
                        Variant("v2", _bras(j, "B", _succes(2))), _taches(2), repeats=3,
                        on_attempt=lambda bras, tache, a: vus.append((bras, tache, a.index)))
        assert len(vus) == 2 * 2 * 3
        assert {b for b, _t, _i in vus} == {"v1", "v2"}


class TestControle:
    def test_deux_bras_identiques_sortent_indistinguables(self) -> None:
        r = compare_configs(_bras_p(0.5, 1), _bras_p(0.5, 2), _taches(), repeats=8, control=True)
        assert r.control is not None and r.control.ok and r.trustworthy
        assert "Contrôle A/A : indistinguable, comme attendu" in r.summary()
        assert r.control.name == "A (contrôle A/A)"

    def test_un_controle_en_echec_invalide_le_verdict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Si la mesure « voit » une différence entre A et sa copie, aucun verdict A/B ne vaut."""
        vrai = cmp._decider
        appels = {"n": 0}

        def truque(pairs: Any, confidence: float) -> Any:
            appels["n"] += 1
            if appels["n"] == 2:                         # 1er appel : A/B ; 2e : A/A
                return (0.4, 0.2, 0.6, 0.001, B_BETTER)
            return vrai(pairs, confidence)

        monkeypatch.setattr(cmp, "_decider", truque)
        j: list[tuple[str, str]] = []
        r = compare_configs(_bras(j, "A", _succes(2)), _bras(j, "B", _succes(7)), _taches(),
                            repeats=8, control=True)
        assert r.control is not None and not r.control.ok
        assert not r.trustworthy
        assert "VERDICT NON FIABLE" in r.summary()
        assert any("Contrôle A/A en échec" in n for n in r.notes)


class TestEmpreintes:
    def _agent(self, systeme: str, outil: bool = False) -> Agent:
        agent = Agent(FakeLLMProvider(["42"]), system_prompt=systeme, max_steps=3)
        if outil:
            @agent.tool
            def lire(x: int) -> dict:
                """Lit x."""
                return {"x": x}
        return agent

    def _tache(self) -> list[EvalTask]:
        return [EvalTask("q", "combien ?", lambda r: "42" in r.output)]

    def test_meme_configuration_meme_empreinte(self) -> None:
        r1 = compare_configs(lambda: self._agent("v1"), lambda: self._agent("v1"), self._tache(),
                             repeats=2)
        r2 = compare_configs(lambda: self._agent("v1"), lambda: self._agent("v1"), self._tache(),
                             repeats=3)
        assert r1.fingerprints["A"] == r1.fingerprints["B"] == r2.fingerprints["A"]
        assert r1.fingerprints["suite"] == r2.fingerprints["suite"]

    def test_un_prompt_un_outil_un_parametre_changent_l_empreinte(self) -> None:
        r = compare_configs(lambda: self._agent("v1"), lambda: self._agent("v2"), self._tache(),
                            repeats=2)
        assert r.fingerprints["A"] != r.fingerprints["B"]
        r = compare_configs(lambda: self._agent("v1"), lambda: self._agent("v1", outil=True),
                            self._tache(), repeats=2)
        assert r.fingerprints["A"] != r.fingerprints["B"]
        r = compare_configs(Variant("a", lambda: self._agent("v1"), {"essai": 1}),
                            Variant("b", lambda: self._agent("v1"), {"essai": 2}),
                            self._tache(), repeats=2)
        assert r.fingerprints["a"] != r.fingerprints["b"]

    def test_la_suite_change_quand_la_consigne_ou_le_juge_changent(self) -> None:
        def run(prompt: str, attendu: str) -> str:
            tache = [EvalTask("q", prompt, lambda r: attendu in r.output)]
            return compare_configs(lambda: self._agent("v1"), lambda: self._agent("v1"), tache,
                                   repeats=2).fingerprints["suite"]

        base = run("combien ?", "42")
        assert run("combien ?", "42") == base
        assert run("combien font 6 x 7 ?", "42") != base
        assert run("combien ?", "41") != base

    def test_une_fabrique_instable_est_signalee(self) -> None:
        etat = {"n": 0}

        def fabrique() -> Agent:
            etat["n"] += 1
            return self._agent(f"v{etat['n']}")           # le prompt change à chaque appel

        r = compare_configs(fabrique, lambda: self._agent("v1"), self._tache(), repeats=3)
        assert any("DIFFÉRENTS" in n for n in r.notes)


class TestCout:
    def test_le_changement_relatif_et_son_intervalle(self) -> None:
        j: list[tuple[str, str]] = []
        r = compare_configs(_bras(j, "A", _succes(5), jetons=100), _bras(j, "B", _succes(5), jetons=60),
                            _taches(), repeats=5)
        assert r.cost is not None
        assert r.cost.tokens_per_attempt_a == pytest.approx(100)
        assert r.cost.tokens_per_attempt_b == pytest.approx(60)
        assert r.cost.relative_change == pytest.approx(-0.4)
        assert r.cost.interval == (pytest.approx(-0.4), pytest.approx(-0.4))   # aucune dispersion
        assert r.cost.tokens_per_success_a == pytest.approx(100)

    def test_le_bootstrap_encadre_un_cout_bruite(self) -> None:
        import random

        rng = random.Random(3)

        def bruite(base: int):
            class _A:
                def run(self, prompt: str, context: Any = None) -> _Res:
                    return _Res(True, base + rng.randrange(0, 60))
            return lambda: _A()

        r = compare_configs(bruite(100), bruite(60), _taches(), repeats=10)
        assert r.cost is not None and r.cost.interval is not None
        bas, haut = r.cost.interval
        assert bas <= r.cost.relative_change <= haut         # type: ignore[operator]
        assert bas < haut and haut < 0                        # B coûte moins, avec une marge

    def test_pas_d_usage_pas_de_cout_jamais_un_zero_invente(self) -> None:
        j: list[tuple[str, str]] = []
        r = compare_configs(_bras(j, "A", _succes(2), jetons=None), _bras(j, "B", _succes(2), jetons=None),
                            _taches(2), repeats=2)
        assert r.cost is None
        assert "Coût" not in r.summary()

    def test_le_cout_par_succes_compte_les_echecs(self) -> None:
        j: list[tuple[str, str]] = []
        r = compare_configs(_bras(j, "A", _succes(2), jetons=100), _bras(j, "B", _succes(4), jetons=100),
                            _taches(), repeats=4)
        assert r.cost is not None
        assert r.cost.tokens_per_success_a == pytest.approx(100 * 16 / 8)   # 16 essais payés, 8 succès
        assert r.cost.tokens_per_success_b == pytest.approx(100 * 16 / 16)

    def test_le_tarif_de_l_hote(self) -> None:
        j: list[tuple[str, str]] = []
        r = compare_configs(_bras(j, "A", _succes(4), jetons=100), _bras(j, "B", _succes(4), jetons=100),
                            _taches(), repeats=4,
                            cost_fn=lambda u: (u.total_tokens or 0) * 0.002)
        assert r.cost is not None
        assert r.cost.cost_per_success_a == pytest.approx(100 * 0.002)


class TestPuissance:
    def test_un_petit_plan_ne_detecte_que_de_gros_ecarts(self) -> None:
        assert detectable_difference(1, 3) is None
        assert detectable_difference(4, 5) is not None and detectable_difference(4, 5) >= 0.35  # type: ignore[operator]

    def test_plus_de_repetitions_plus_de_finesse(self) -> None:
        vus = [detectable_difference(4, n) for n in (5, 8, 12)]
        assert all(v is not None for v in vus)
        assert vus[0] >= vus[1] >= vus[2]                      # type: ignore[operator]
        assert vus[0] > vus[2]                                 # type: ignore[operator]

    def test_deterministe_et_valide(self) -> None:
        assert detectable_difference(4, 8, seed=5) == detectable_difference(4, 8, seed=5)
        with pytest.raises(ValueError):
            detectable_difference(0, 5)
        with pytest.raises(ValueError):
            detectable_difference(2, 5, power=1.0)

    def test_le_rapport_donne_la_meme_valeur(self) -> None:
        j: list[tuple[str, str]] = []
        r = compare_configs(_bras(j, "A", _succes(4)), _bras(j, "B", _succes(4)), _taches(),
                            repeats=8, seed=11)
        assert r.detectable == detectable_difference(4, 8, seed=11)


class TestSorties:
    def test_to_dict_est_json_safe_et_complet(self) -> None:
        j: list[tuple[str, str]] = []
        r = compare_configs(_bras_p(0.0, 1), _bras(j, "B", _succes(7)), _taches(),
                            repeats=8, control=True)
        d = json.loads(json.dumps(r.to_dict()))
        assert d["verdict"] == B_BETTER and d["trustworthy"] is True
        assert d["control"]["ok"] is True
        assert len(d["tasks"]) == 4
        assert set(d["reports"]) == {"A", "B", "A (contrôle A/A)"}
        assert d["reports"]["A"]["t1"]["successes"] == 0
        assert d["reports"]["B"]["t1"]["successes"] == 7
        assert set(d["fingerprints"]) >= {"suite", "A", "B"}

    def test_summary_donne_les_chiffres_et_les_limites(self) -> None:
        j: list[tuple[str, str]] = []
        r = compare_configs(Variant("v1", _bras(j, "A", _succes(2))),
                            Variant("v2", _bras(j, "B", _succes(7))), _taches(), repeats=8)
        s = r.summary()
        assert "A « v1 » contre B « v2 »" in s
        assert "4 tâche(s) x 8 répétition(s) par bras" in s
        assert "intervalle approché à 95%" in s
        assert "test exact p" in s
        assert "Empreintes (structure déclarée, pas le code des outils)" in s

    def test_les_agents_reels_marchent_aussi(self) -> None:
        """Pas seulement des factices : de vrais `Agent`, un vrai juge."""
        bon = lambda: Agent(FakeLLMProvider(["42"]), max_steps=3)     # noqa: E731
        mauvais = lambda: Agent(FakeLLMProvider(["je ne sais pas"]), max_steps=3)  # noqa: E731
        taches = [EvalTask(f"q{i}", "combien ?", lambda r: "42" in r.output) for i in range(1, 5)]
        r = compare_configs(mauvais, bon, taches, repeats=6)
        assert r.verdict == B_BETTER
        assert r.rate_a == 0.0 and r.rate_b == 1.0
