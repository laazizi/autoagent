"""`cascade` — le modèle bon marché d'abord, le gros si TON juge dit non (0.21.0).

Ce que ces tests verrouillent :

  1. ON S'ARRÊTE AU PREMIER PALIER ACCEPTÉ — le gros modèle n'est pas appelé
     quand le petit a convaincu le juge.
  2. LES PALIERS RATÉS SE PAIENT — la dépense est cumulée sur tout ce qui a été
     essayé, pas seulement sur le palier qui a répondu.
  3. UNE PAUSE D'APPROBATION N'EST PAS UN ÉCHEC — `ApprovalRequired` remonte ;
     monter de palier dessus contournerait le feu vert humain par un autre
     modèle. C'est le test le plus important du fichier.
  4. Un palier qui plante est un palier raté, la cascade continue ; un juge qui
     lève refuse ; aucun palier accepté → `accepted` faux, dernier résultat rendu.
"""

from __future__ import annotations

import pytest

from autoagent import ApprovalRequired, cascade
from autoagent.errors import MaxStepsExceeded
from autoagent.schema import TokenUsage


class Resultat:
    def __init__(self, output: str, entree: int, sortie: int, steps: int = 1) -> None:
        self.output = output
        self.steps = steps
        self.usage = TokenUsage(input_tokens=entree, output_tokens=sortie,
                                total_tokens=entree + sortie)
        self.messages: list = []


class Palier:
    """Un agent scripté : rend `sortie`, ou lève `erreur`. Compte ses appels."""

    def __init__(self, modele: str, sortie: str = "", entree: int = 100, sortie_jetons: int = 20,
                 erreur: Exception | None = None) -> None:
        class _Cfg:
            model = modele

        class _Prov:
            config = _Cfg()

        self.provider = _Prov()
        self._sortie, self._entree, self._sortie_j, self._erreur = sortie, entree, sortie_jetons, erreur
        self.appels = 0

    def run(self, prompt, context=None):  # type: ignore[no-untyped-def]
        self.appels += 1
        if self._erreur is not None:
            raise self._erreur
        return Resultat(self._sortie, self._entree, self._sortie_j)


def juge_ok_si_42(res) -> bool:  # type: ignore[no-untyped-def]
    return "42" in res.output


class TestArretAuPremierAccepte:
    def test_le_gros_n_est_pas_appele_si_le_petit_convainc(self) -> None:
        petit, gros = Palier("lite", "la réponse est 42"), Palier("pro", "42")
        r = cascade([petit, gros], "q", check=juge_ok_si_42)
        assert r.accepted and r.tier == 1 and r.escalations == 0
        assert (petit.appels, gros.appels) == (1, 0)
        assert r.result.output == "la réponse est 42"

    def test_on_monte_quand_le_juge_refuse(self) -> None:
        petit, gros = Palier("lite", "je ne sais pas"), Palier("pro", "c'est 42")
        r = cascade([petit, gros], "q", check=juge_ok_si_42)
        assert r.tier == 2 and r.escalations == 1
        assert (petit.appels, gros.appels) == (1, 1)
        assert [a.ok for a in r.attempts] == [False, True]
        assert [a.model for a in r.attempts] == ["lite", "pro"]


class TestLesPaliersRatesSePaient:
    def test_la_depense_cumule_le_palier_rate(self) -> None:
        petit = Palier("lite", "non", entree=1000, sortie_jetons=50)
        gros = Palier("pro", "42", entree=4000, sortie_jetons=200)
        r = cascade([petit, gros], "q", check=juge_ok_si_42)
        assert r.usage.input_tokens == 5000 and r.usage.output_tokens == 250
        assert r.usage.total_tokens == 5250
        assert "5250 jetons" in r.summary()

    def test_un_palier_qui_plante_avec_etat_porte_sa_depense(self) -> None:
        class Etat:
            input_tokens, output_tokens = 300, 30

        exc = MaxStepsExceeded("trop d'étapes")
        exc.state = Etat()  # type: ignore[attr-defined]
        petit, gros = Palier("lite", erreur=exc), Palier("pro", "42", entree=100, sortie_jetons=10)
        r = cascade([petit, gros], "q", check=juge_ok_si_42)
        assert r.tier == 2
        assert r.attempts[0].error.startswith("MaxStepsExceeded")
        assert r.usage.total_tokens == 330 + 110


class TestUnePauseNEstPasUnEchec:
    def test_approval_required_remonte_sans_monter_de_palier(self) -> None:
        pause = ApprovalRequired("un humain doit valider")
        petit, gros = Palier("lite", erreur=pause), Palier("pro", "42")
        with pytest.raises(ApprovalRequired):
            cascade([petit, gros], "q", check=juge_ok_si_42)
        assert gros.appels == 0, "la cascade a contourné le feu vert humain avec un autre modèle"


class TestRobustesse:
    def test_un_juge_qui_leve_refuse(self) -> None:
        def juge_casse(res) -> bool:  # type: ignore[no-untyped-def]
            raise RuntimeError("juge cassé")

        petit, gros = Palier("lite", "42"), Palier("pro", "42")
        r = cascade([petit, gros], "q", check=juge_casse)
        assert not r.accepted and all("check raised" in a.error for a in r.attempts)

    def test_aucun_palier_accepte_rend_le_dernier(self) -> None:
        r = cascade([Palier("lite", "a"), Palier("pro", "b")], "q", check=juge_ok_si_42)
        assert not r.accepted and r.tier is None and r.result.output == "b"
        assert r.summary().startswith("aucun palier accepté")

    def test_fabriques_acceptees(self) -> None:
        crees: list[Palier] = []

        def fabrique() -> Palier:
            p = Palier("pro", "42")
            crees.append(p)
            return p

        r = cascade([fabrique], "q", check=juge_ok_si_42)
        assert r.accepted and len(crees) == 1

    def test_sans_palier_ou_sans_juge(self) -> None:
        with pytest.raises(ValueError):
            cascade([], "q", check=juge_ok_si_42)
        with pytest.raises(TypeError):
            cascade([Palier("x", "42")], "q", check=None)  # type: ignore[arg-type]

    def test_on_tier_est_appele_et_fail_open(self) -> None:
        vus: list[int] = []

        def rappel(a) -> None:  # type: ignore[no-untyped-def]
            vus.append(a.index)
            raise RuntimeError("le rappel plante, la cascade non")

        r = cascade([Palier("lite", "non"), Palier("pro", "42")], "q",
                    check=juge_ok_si_42, on_tier=rappel)
        assert vus == [1, 2] and r.tier == 2
