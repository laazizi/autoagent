"""Eval à coût normalisé (0.21.0) — « score à dépense fixe », pas score tout seul.

Un pass^k isolé ne dit pas ce qu'il a coûté. Ces tests verrouillent :

  1. LA DÉPENSE EST CUMULÉE SUR TOUTES LES TENTATIVES, ratées comprises — les
     échecs se paient aussi.
  2. AUCUN CHIFFRE INVENTÉ : sans usage rapporté → None, pas zéro ; zéro succès
     → None, pas un infini.
  3. LE TARIF VIENT DE L'HÔTE (`cost_fn`) — jamais de la lib.
  4. `pass_hat_k_at_budget` : une réussite qui crève le budget ne compte pas, et
     une tentative sans usage ne peut pas prouver qu'elle est dessous.
"""

from __future__ import annotations

from autoagent.eval import Attempt, ReliabilityReport, run_k
from autoagent.schema import TokenUsage


def _rapport(*tentatives: tuple[bool, int | None, int | None]) -> ReliabilityReport:
    """(ok, entrée, sortie) par tentative ; entrée None = usage non rapporté."""
    r = ReliabilityReport(k=len(tentatives))
    for i, (ok, entree, sortie) in enumerate(tentatives, 1):
        a = Attempt(index=i, ok=ok)
        if entree is not None:
            a.input_tokens, a.output_tokens = entree, sortie or 0
            a.total_tokens = entree + (sortie or 0)
        r.attempts.append(a)
    return r


class TestDepenseCumulee:
    def test_les_echecs_se_paient_aussi(self) -> None:
        r = _rapport((True, 1000, 100), (False, 3000, 50), (True, 1000, 100))
        assert r.usage.total_tokens == 5250
        assert r.tokens_per_success == 5250 / 2

    def test_cache_agrege_seulement_s_il_est_rapporte(self) -> None:
        r = _rapport((True, 100, 10), (True, 100, 10))
        assert r.usage.cached_tokens is None
        r.attempts[0].cached_tokens = 40
        assert r.usage.cached_tokens == 40


class TestAucunChiffreInvente:
    def test_sans_usage_tout_est_none(self) -> None:
        r = _rapport((True, None, None), (True, None, None))
        assert r.usage is None
        assert r.tokens_per_success is None
        assert r.cost(lambda u: 1.0) is None
        assert "jetons/succès" not in r.summary()

    def test_zero_succes_pas_d_infini(self) -> None:
        r = _rapport((False, 500, 10), (False, 500, 10))
        assert r.usage.total_tokens == 1020
        assert r.tokens_per_success is None
        assert r.cost_per_success(lambda u: 1.0) is None


class TestTarifDeLHote:
    def test_cost_fn_recoit_le_token_usage_agrege(self) -> None:
        r = _rapport((True, 1000, 100), (True, 1000, 100))
        recu: list[TokenUsage] = []

        def tarif(u: TokenUsage) -> float:
            recu.append(u)
            return (u.input_tokens * 0.30 + u.output_tokens * 2.50) / 1_000_000

        total = r.cost(tarif)
        assert recu[0].input_tokens == 2000 and recu[0].output_tokens == 200
        assert abs(total - (2000 * 0.30 + 200 * 2.50) / 1e6) < 1e-12
        assert abs(r.cost_per_success(tarif) - total / 2) < 1e-12


class TestBudgetFixe:
    def test_toutes_reussies_sous_le_budget(self) -> None:
        r = _rapport((True, 800, 100), (True, 700, 100))
        assert r.pass_hat_k_at_budget(1000) == 1.0

    def test_une_reussite_qui_creve_le_budget_ne_compte_pas(self) -> None:
        r = _rapport((True, 800, 100), (True, 1500, 100))
        assert r.pass_hat_k == 1.0, "sans budget, tout a réussi"
        assert r.pass_hat_k_at_budget(1000) == 0.0, "avec budget, la 2e est hors jeu"

    def test_sans_usage_rapporte_on_ne_peut_pas_prouver_le_budget(self) -> None:
        r = _rapport((True, None, None))
        assert r.pass_hat_k == 1.0
        assert r.pass_hat_k_at_budget(10_000) == 0.0

    def test_rapport_vide(self) -> None:
        assert ReliabilityReport(k=3).pass_hat_k_at_budget(1) == 0.0


class TestRunKRemplitLeDetail:
    def test_les_champs_detailles_arrivent_depuis_le_resultat(self) -> None:
        class Res:
            steps = 2
            output = "ok"
            usage = TokenUsage(input_tokens=300, output_tokens=40, cached_tokens=120)

        class Ag:
            def run(self, prompt, context=None):  # type: ignore[no-untyped-def]
                return Res()

        r = run_k(Ag(), "x", k=2, check=lambda res: True)
        a = r.attempts[0]
        assert (a.input_tokens, a.output_tokens, a.cached_tokens, a.total_tokens) == (300, 40, 120, 340)
        d = r.to_dict()
        assert d["total_tokens"] == 680 and d["tokens_per_success"] == 340.0
        assert d["attempts"][0]["cached_tokens"] == 120
        assert "340 jetons/succès" in r.summary()
