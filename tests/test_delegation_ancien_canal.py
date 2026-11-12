"""L'ANCIEN canal de dépense (`handler.__autoagent_usage__`, 0.19.0 à 0.21.0) reste honoré.

En 0.19.0, `Agent.as_tool()` posait la dépense du sous-agent sur un attribut de
la fonction-outil, `__autoagent_usage__`, que la boucle du parent lisait puis
vidait. La 0.22.0 l'a remplacé par un canal par thread
(`__autoagent_usage_channel__`) — et a cessé, sans le dire, de poser et de lire
l'ancien attribut.

Un hôte qui ENVELOPPE un outil de délégation dans sa propre fonction (pour borner
le spécialiste, ajouter un journal, traduire une erreur) n'avait qu'un moyen de
faire remonter la dépense : recopier cet attribut sur son enveloppe. Constat du
12 nov. 2026 sur un projet consommateur écrit contre la 0.21.0 : avec une
bibliothèque plus récente, la dépense de ses spécialistes ne remontait plus —
sans erreur, `token_budget` redevenait aveugle à la délégation.

Ces tests fixent le contrat rétabli : l'outil de délégation expose toujours
l'attribut, et la boucle le lit sur un outil qui n'a pas le canal par thread.
"""

from __future__ import annotations

from typing import Any

import pytest

from autoagent import Agent, MaxStepsExceeded, TokenBudgetExceeded, delegate_to
from autoagent.providers.base import LLMProvider
from autoagent.schema import LLMResponse, ModelConfig, TokenUsage, ToolCall


class Fournisseur(LLMProvider):
    """Rend les réponses prévues, dans l'ordre."""

    def __init__(self, reponses: list[LLMResponse]) -> None:
        super().__init__(ModelConfig(provider="f", model="f", api_key="x"))
        self.reponses = list(reponses)

    def complete(self, request):  # type: ignore[no-untyped-def]
        return self.reponses.pop(0)


def _reponse(entree: int, sortie: int, *, appel: str | None = None,
             arguments: dict[str, Any] | None = None) -> LLMResponse:
    calls = [ToolCall(id="c1", name=appel, arguments=arguments or {"demande": "x"})] if appel else []
    return LLMResponse(content="" if appel else "fini", tool_calls=calls,
                       usage=TokenUsage(input_tokens=entree, output_tokens=sortie))


def _enveloppe(fabrique_du_specialiste):  # type: ignore[no-untyped-def]
    """L'enveloppe d'un hôte, telle qu'écrite contre la 0.21.0 : un spécialiste neuf par appel,
    et la dépense recopiée de l'outil interne sur l'enveloppe."""

    def confier(demande: str) -> dict:
        """Confie la demande au spécialiste."""
        confier.__autoagent_usage__ = None  # type: ignore[attr-defined]
        interne = fabrique_du_specialiste().as_tool(name="confier", description="délègue")
        rendu = interne(demande)
        confier.__autoagent_usage__ = getattr(interne, "__autoagent_usage__", None)  # type: ignore[attr-defined]
        rendu["journal"] = ["fait"]
        return rendu

    confier.__autoagent_usage__ = None  # type: ignore[attr-defined]
    return confier


def _specialiste() -> Agent:
    return Agent(Fournisseur([_reponse(1000, 500)]))


def _parent(outil, *, budget: int | None = None, appels: int = 1, **kwargs) -> Agent:  # type: ignore[no-untyped-def]
    reponses = [_reponse(100, 10, appel="confier") for _ in range(appels)] + [_reponse(120, 5)]
    parent = Agent(Fournisseur(reponses), max_steps=appels + 3, token_budget=budget, **kwargs)
    parent.add_tool(outil)
    return parent


class TestLOutilDeDelegationExposeToujoursLAttribut:
    def test_as_tool_pose_la_depense_sur_la_fonction(self) -> None:
        outil = _specialiste().as_tool(name="expert", description="délègue")
        assert outil.__autoagent_usage__ is None            # présent dès la création, vide
        outil("vas-y")
        assert outil.__autoagent_usage__ == TokenUsage(input_tokens=1000, output_tokens=500)

    def test_un_appel_rate_ne_laisse_pas_la_depense_du_precedent(self) -> None:
        """Remis à zéro au DÉBUT de chaque appel : la dépense d'un appel réussi ne doit pas être
        relue après un appel qui échoue sans rien rapporter."""
        class Casse(LLMProvider):
            def __init__(self) -> None:
                super().__init__(ModelConfig(provider="f", model="f", api_key="x"))
                self.tour = 0

            def complete(self, request):  # type: ignore[no-untyped-def]
                self.tour += 1
                if self.tour == 1:
                    return _reponse(1000, 500)
                raise RuntimeError("panne du fournisseur")

        outil = Agent(Casse()).as_tool(name="expert", description="délègue")
        outil("premier")
        assert outil.__autoagent_usage__ is not None
        with pytest.raises(Exception, match="panne du fournisseur"):
            outil("second")
        assert outil.__autoagent_usage__ is None

    def test_un_sous_agent_qui_epuise_ses_pas_rapporte_sa_depense(self) -> None:
        """Depuis la 0.22.0 un échec rapporte ce qu'il a coûté ; l'ancien attribut le reflète aussi."""
        boucle = Agent(Fournisseur([_reponse(300, 30, appel="rien"), _reponse(300, 30, appel="rien")]),
                       max_steps=2)

        @boucle.add_tool
        def rien(demande: str) -> str:
            """Ne fait rien."""
            return "rien"

        outil = boucle.as_tool(name="expert", description="délègue")
        with pytest.raises(MaxStepsExceeded):
            outil("vas-y")
        assert outil.__autoagent_usage__ == TokenUsage(input_tokens=600, output_tokens=60)

    def test_delegate_to_pose_aussi_la_depense(self) -> None:
        outil = delegate_to({"a": _specialiste()})
        assert outil.__autoagent_usage__ is None
        outil([{"specialist": "a", "request": "x"}])
        assert outil.__autoagent_usage__ == TokenUsage(input_tokens=1000, output_tokens=500)

    def test_enregistre_directement_rien_n_est_compte_deux_fois(self) -> None:
        """L'outil de délégation porte les DEUX canaux : la boucle n'en lit qu'un."""
        parent = Agent(Fournisseur([_reponse(100, 10, appel="expert", arguments={"request": "x"}),
                                    _reponse(120, 5)]), max_steps=4)
        outil = parent.add_tool(_specialiste().as_tool(name="expert", description="délègue"))
        usage = parent.run("vas-y").usage
        assert (usage.input_tokens, usage.output_tokens) == (1220, 515)
        assert outil.__autoagent_usage__ is None            # consommé, comme en 0.21.0


class TestLaBoucleLitLAncienAttributSurUneEnveloppe:
    def test_la_depense_du_specialiste_enveloppe_est_comptee(self) -> None:
        usage = _parent(_enveloppe(_specialiste)).run("vas-y").usage
        # parent 100+120 / 10+5, spécialiste 1000 / 500
        assert (usage.input_tokens, usage.output_tokens) == (1220, 515)

    def test_le_plafond_voit_la_delegation_enveloppee(self) -> None:
        """Le cœur du sujet : sans la lecture de l'ancien attribut, ce plafond ne se déclenchait plus."""
        with pytest.raises(TokenBudgetExceeded):
            _parent(_enveloppe(_specialiste), budget=500).run("vas-y")

    def test_la_depense_est_consommee(self) -> None:
        outil = _enveloppe(_specialiste)
        _parent(outil).run("vas-y")
        assert outil.__autoagent_usage__ is None

    def test_deux_appels_sont_comptes_une_fois_chacun(self) -> None:
        usage = _parent(_enveloppe(_specialiste), appels=2).run("vas-y").usage
        assert (usage.input_tokens, usage.output_tokens) == (100 + 100 + 120 + 2000, 10 + 10 + 5 + 1000)

    def test_une_depense_posee_par_l_hote_lui_meme_est_comptee(self) -> None:
        """L'hôte peut poser lui-même un `TokenUsage` (ce qu'il a lu dans le `state` d'un arrêt sur borne)."""
        def confier(demande: str) -> dict:
            """Confie la demande."""
            confier.__autoagent_usage__ = TokenUsage(input_tokens=700, output_tokens=70)  # type: ignore[attr-defined]
            return {"echec": True}

        confier.__autoagent_usage__ = None  # type: ignore[attr-defined]
        usage = _parent(confier).run("vas-y").usage
        assert (usage.input_tokens, usage.output_tokens) == (920, 85)

    def test_une_valeur_qui_n_est_pas_une_depense_est_ignoree(self) -> None:
        """Seul un `TokenUsage` entre dans la comptabilité : une valeur quelconque ne casse pas le run."""
        def confier(demande: str) -> str:
            """Confie la demande."""
            confier.__autoagent_usage__ = "beaucoup"  # type: ignore[attr-defined]
            return "ok"

        usage = _parent(confier).run("vas-y").usage
        assert (usage.input_tokens, usage.output_tokens) == (220, 15)
        assert confier.__autoagent_usage__ is None  # type: ignore[attr-defined]

    def test_une_methode_liee_ne_casse_pas_le_run(self) -> None:
        """Un attribut ne se pose pas sur une méthode liée : il se vide sur sa fonction."""
        class Hote:
            def confier(self, demande: str) -> str:
                """Confie la demande."""
                Hote.confier.__autoagent_usage__ = TokenUsage(input_tokens=700, output_tokens=70)  # type: ignore[attr-defined]
                return "ok"

        usage = _parent(Hote().confier, appels=2).run("vas-y").usage
        assert (usage.input_tokens, usage.output_tokens) == (100 + 100 + 120 + 1400, 10 + 10 + 5 + 140)
        assert Hote.confier.__autoagent_usage__ is None  # type: ignore[attr-defined]

    def test_un_outil_ordinaire_ne_compte_rien(self) -> None:
        def confier(demande: str) -> str:
            """Un outil sans dépense."""
            return "ok"

        usage = _parent(confier).run("vas-y").usage
        assert (usage.input_tokens, usage.output_tokens) == (220, 15)


class TestUneEnveloppeNEstPasUneFonctionPourLeCode:
    def test_l_enveloppe_n_est_pas_exposee_au_code_du_modele(self) -> None:
        """Comme un sous-agent : appelée depuis le code du modèle, sa dépense ne serait pas comptée."""
        agent = Agent(Fournisseur([]))
        agent.add_tool(_enveloppe(_specialiste))

        @agent.add_tool
        def lire(chemin: str) -> str:
            """Lit un fichier."""
            return "contenu"

        fonctions = agent._outils_pour_le_code(None)
        assert "confier" in agent.registry
        assert "confier" not in fonctions
        assert "lire" in fonctions
