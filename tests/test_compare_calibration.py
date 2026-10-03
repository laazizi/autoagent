"""Calibration de `compare_configs` : ce que la règle de verdict garantit, MESURÉ.

Un comparateur qui se trompe sans le dire est pire que pas de comparateur. Ce fichier
ne vérifie pas du code, il vérifie des PROMESSES statistiques, par énumération
exacte quand c'est possible, par simulation à graine fixe sinon.

La mesure qui a décidé de la conception (écrite avant le module, refaite ici sur
l'implémentation livrée) : l'intervalle de Wilson-Newcombe, pris seul comme règle de
verdict, déclarait un gagnant alors que A et B étaient IDENTIQUES dans jusqu'à 14,6 %
des cas (2 tâches x 3 essais, p = 0,5) et 11,5 % (2 x 5) au lieu des 5 % annoncés
— et la moyenne sur des suites variées (96-99 % de couverture) cachait ces cas.
Le test exact par permutation stratifié ne dépasse pas 5 % par construction ; la
règle livrée (les DEUX d'accord) est encore plus prudente.
"""

from __future__ import annotations

import itertools
import math
import random

import pytest

from autoagent import compare as cmp
from autoagent.compare import detectable_difference, paired_interval, paired_p_value

ALPHA = 0.05


def _proba(t: int, n: int, pa: float, pb: float, regle) -> float:
    """P(la règle déclare une différence), par ÉNUMÉRATION de toutes les issues possibles :
    t tâches, n essais par bras, succès binomiaux. Exact, aucun tirage."""
    pmf_a = [math.comb(n, s) * pa ** s * (1 - pa) ** (n - s) for s in range(n + 1)]
    pmf_b = [math.comb(n, s) * pb ** s * (1 - pb) ** (n - s) for s in range(n + 1)]
    total = 0.0
    for issue in itertools.product(range(n + 1), repeat=2 * t):
        pairs = [(issue[2 * i], n, issue[2 * i + 1], n) for i in range(t)]
        prob = 1.0
        for i in range(t):
            prob *= pmf_a[issue[2 * i]] * pmf_b[issue[2 * i + 1]]
        if prob and regle(pairs):
            total += prob
    return total


def _intervalle_seul(pairs: list[tuple[int, int, int, int]]) -> bool:
    _delta, bas, haut = paired_interval(pairs)
    return bas > 0 or haut < 0


def _test_exact_seul(pairs: list[tuple[int, int, int, int]]) -> bool:
    return paired_p_value(pairs) < ALPHA


def _regle_livree(pairs: list[tuple[int, int, int, int]]) -> bool:
    return cmp._decider(pairs, 1 - ALPHA)[4] != cmp.INDISTINGUISHABLE


def _binom(rng: random.Random, n: int, p: float) -> int:
    return sum(rng.random() < p for _ in range(n))


PLANS = [(1, 5), (2, 3), (2, 5), (3, 3), (3, 4)]


class TestFaussesAlertes:
    """A et B IDENTIQUES : combien de fois la règle déclare-t-elle quand même un gagnant ?"""

    @pytest.mark.parametrize("t,n", PLANS)
    @pytest.mark.parametrize("p", [0.2, 0.5])
    def test_la_regle_livree_ne_depasse_jamais_alpha(self, t: int, n: int, p: float) -> None:
        assert _proba(t, n, p, p, _regle_livree) <= ALPHA + 1e-12

    @pytest.mark.parametrize("t,n", PLANS)
    def test_le_test_exact_ne_depasse_pas_alpha_par_construction(self, t: int, n: int) -> None:
        assert _proba(t, n, 0.5, 0.5, _test_exact_seul) <= ALPHA + 1e-12

    def test_l_intervalle_seul_depasserait_alpha_c_est_pourquoi_il_ne_decide_pas_seul(self) -> None:
        """La mesure qui justifie la conception. Si quelqu'un « simplifie » la règle en ne
        gardant que l'intervalle, ces chiffres redeviennent ceux d'une règle fausse."""
        assert _proba(2, 3, 0.5, 0.5, _intervalle_seul) > 0.14     # mesuré : 0,146
        assert _proba(2, 5, 0.5, 0.5, _intervalle_seul) > 0.11     # mesuré : 0,115
        assert _proba(1, 5, 0.5, 0.5, _intervalle_seul) > ALPHA    # mesuré : 0,061
        # … alors que la règle livrée, sur les mêmes configurations identiques :
        assert _proba(2, 3, 0.5, 0.5, _regle_livree) < 0.02        # mesuré : 0,012

    def test_a_et_b_se_confondent_sous_etiquettes_echangees(self) -> None:
        """L'hypothèse du test exact : échanger les étiquettes A/B ne change pas la p-value."""
        rng = random.Random(5)
        for _ in range(200):
            pairs = [(rng.randrange(6), 5, rng.randrange(6), 5) for _ in range(3)]
            inverse = [(sb, nb, sa, na) for sa, na, sb, nb in pairs]
            assert paired_p_value(pairs) == pytest.approx(paired_p_value(inverse))


class TestCouverture:
    """L'intervalle est APPROCHÉ — la doc le dit, ces tests le mesurent."""

    @staticmethod
    def _couverture(t: int, n: int, pa: float, pb: float, sims: int, graine: int) -> float:
        rng = random.Random(graine)
        vrai = pb - pa
        ok = 0
        for _ in range(sims):
            pairs = [(_binom(rng, n, pa), n, _binom(rng, n, pb), n) for _ in range(t)]
            _d, bas, haut = paired_interval(pairs)
            ok += bas <= vrai <= haut
        return ok / sims

    def test_sur_des_suites_variees_la_couverture_moyenne_est_proche_de_95(self) -> None:
        grille = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        for t, n in ((4, 5), (4, 8)):
            rng = random.Random(900 + t * 10 + n)
            sims = 1200
            ok = 0
            for _ in range(sims):
                pairs = []
                vrai = 0.0
                for _t in range(t):
                    pa, pb = rng.choice(grille), rng.choice(grille)
                    vrai += pb - pa
                    pairs.append((_binom(rng, n, pa), n, _binom(rng, n, pb), n))
                _d, bas, haut = paired_interval(pairs)
                ok += bas <= vrai / t <= haut
            assert ok / sims >= 0.94                              # mesuré : 0,96-0,97

    def test_le_pire_cas_connu_reste_au_dessus_de_la_borne_annoncee(self) -> None:
        """Taux extrêmes, beaucoup de tâches, peu d'essais : la couverture tombe vers 83 %.
        C'est la borne basse que l'en-tête du module annonce ; elle ne doit pas s'effondrer."""
        c = self._couverture(8, 5, 0.95, 0.1, 1000, 77)
        assert 0.78 <= c <= 0.93, c                               # mesuré : 0,83


class TestPuissance:
    """Ce que le plan voit — et ne voit pas."""

    @staticmethod
    def _puissance(t: int, n: int, pa: float, pb: float, sims: int = 800, graine: int = 42) -> float:
        rng = random.Random(graine)
        gagne = 0
        for _ in range(sims):
            pairs = [(_binom(rng, n, pa), n, _binom(rng, n, pb), n) for _ in range(t)]
            gagne += cmp._decider(pairs, 1 - ALPHA)[4] == cmp.B_BETTER
        return gagne / sims

    def test_un_ecart_net_sur_un_plan_correct_est_detecte(self) -> None:
        assert self._puissance(8, 8, 0.5, 0.8) >= 0.85            # mesuré : 0,95
        assert self._puissance(4, 8, 0.2, 0.8) >= 0.95            # mesuré : 1,00

    def test_un_petit_plan_rate_un_ecart_moyen(self) -> None:
        """4 tâches x 5 essais, 30 points d'écart : on le rate plus d'une fois sur deux.
        C'est exactement la fausse précision que `detectable_difference` dénonce."""
        assert self._puissance(4, 5, 0.5, 0.8) <= 0.55            # mesuré : 0,40

    def test_detectable_difference_est_coherent_avec_la_puissance_mesuree(self) -> None:
        d = detectable_difference(4, 8)
        assert d is not None
        # À l'écart annoncé, la puissance approche la cible ; deux pas de grille plus bas, non.
        # (Plan d'échantillons différent de celui de la fonction : marge de simulation.)
        assert self._puissance(4, 8, 0.5, 0.5 + d, graine=7) >= 0.70
        assert self._puissance(4, 8, 0.5, 0.5 + d - 0.2, graine=7) < 0.80
