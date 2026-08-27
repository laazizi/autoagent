"""La dépense d'un SOUS-AGENT remonte au parent (0.19.0).

`Agent.as_tool()` fait d'un agent l'outil d'un autre. Mais un sous-agent n'est
pas une fonction Python : il part faire son propre run, avec ses propres appels
au LLM et ses propres jetons.

Trou avéré avant ce correctif : ce coût était écrit dans le résultat rendu au
MODÈLE (`payload["tokens"]`, donc lisible dans le transcript) puis jeté. La
boucle du parent n'additionnait que SES propres réponses, si bien que
`result.usage` sous-évaluait le run et que `token_budget` ne voyait rien
passer — un plafond qu'il suffisait de contourner en déléguant.

Le réflexe existait déjà à côté : la compaction mémoire appelle son propre LLM
et son coût est compté depuis la 0.17 (`memory.last_usage`). C'était le même
oubli, au même endroit, pour l'autre sous-appel.
"""

from __future__ import annotations

from autoagent import Agent, TokenBudgetExceeded
from autoagent.providers.base import LLMProvider
from autoagent.schema import LLMResponse, ModelConfig, TokenUsage, ToolCall


class Fournisseur(LLMProvider):
    """Rend les réponses prévues, dans l'ordre."""

    def __init__(self, reponses: list[LLMResponse]) -> None:
        super().__init__(ModelConfig(provider="f", model="f", api_key="x"))
        self.reponses = list(reponses)

    def complete(self, request):  # type: ignore[no-untyped-def]
        return self.reponses.pop(0)


def _reponse(entree: int, sortie: int, *, cache: int | None = None,
             appel: str | None = None) -> LLMResponse:
    calls = ([ToolCall(id="c1", name=appel, arguments={"request": "x"})]
             if appel else [])
    return LLMResponse(
        content="" if appel else "fini",
        tool_calls=calls,
        usage=TokenUsage(input_tokens=entree, output_tokens=sortie, cached_tokens=cache),
    )


def _specialiste(entree: int = 1000, sortie: int = 500,
                 cache: int | None = None) -> Agent:
    return Agent(Fournisseur([_reponse(entree, sortie, cache=cache)]))


def _superviseur(sous: Agent, *, budget: int | None = None,
                 nom: str = "expert", **kwargs) -> Agent:
    parent = Agent(
        Fournisseur([_reponse(100, 10, appel=nom), _reponse(120, 5)]),
        max_steps=4, token_budget=budget, **kwargs)
    parent.add_tool(sous.as_tool(name=nom, description="délègue au spécialiste"))
    return parent


class TestLeCoutRemonte:
    def test_le_run_delegue_est_compte(self) -> None:
        usage = _superviseur(_specialiste()).run("vas-y").usage
        # parent 100+120 / 10+5, spécialiste 1000 / 500
        assert usage.input_tokens == 1220
        assert usage.output_tokens == 515
        assert usage.total_tokens == 1735

    def test_sans_delegation_rien_ne_change(self) -> None:
        """Non-régression : un run ordinaire compte exactement comme avant."""
        agent = Agent(Fournisseur([_reponse(100, 10)]), max_steps=2)
        usage = agent.run("vas-y").usage
        assert (usage.input_tokens, usage.output_tokens) == (100, 10)

    def test_le_chiffre_reste_lisible_par_le_modele(self) -> None:
        """Le total du sous-agent doit RESTER dans le résultat d'outil : il
        renseigne le modèle. Le canal de comptabilité s'ajoute, il ne remplace
        pas."""
        resultat = _superviseur(_specialiste()).run("vas-y")
        outil = next(m for m in resultat.messages if m.role == "tool")
        assert '"tokens": 1500' in outil.content

    def test_un_sous_agent_muet_ne_casse_rien(self) -> None:
        """Un fournisseur qui ne rapporte aucun usage ne doit ni planter ni
        inventer un chiffre."""
        muet = Agent(Fournisseur([LLMResponse(content="ok")]))
        usage = _superviseur(muet).run("vas-y").usage
        assert (usage.input_tokens, usage.output_tokens) == (220, 15)

    def test_la_part_de_cache_du_sous_agent_remonte(self) -> None:
        usage = _superviseur(_specialiste(cache=800)).run("vas-y").usage
        assert usage.cached_tokens == 800


class TestLePlafondSeDeclenche:
    """Le cœur du correctif : un plafond qui ne voyait pas la délégation
    n'était pas un plafond."""

    def test_arret_net_sur_une_depense_deleguee(self) -> None:
        parent = _superviseur(_specialiste(), budget=500)
        try:
            parent.run("vas-y")
        except TokenBudgetExceeded as exc:
            assert exc.spent == 1610, "le compte doit inclure les 1 500 délégués"
            assert exc.state is not None, "l'arrêt reste reprenable"
        else:
            raise AssertionError("le plafond n'a pas vu la dépense du sous-agent")

    def test_un_budget_large_laisse_finir(self) -> None:
        resultat = _superviseur(_specialiste(), budget=50_000).run("vas-y")
        assert resultat.output.strip() == "fini"


class TestPasDeDoubleFacturation:
    def test_deux_appels_ne_refacturent_pas_le_premier(self) -> None:
        """La dépense est CONSOMMÉE à la lecture : sans ça, le deuxième appel
        du même outil réajouterait le coût du premier."""
        sous = Agent(Fournisseur([_reponse(1000, 500), _reponse(200, 100)]))
        parent = Agent(Fournisseur([
            _reponse(100, 10, appel="expert"),
            _reponse(100, 10, appel="expert"),
            _reponse(120, 5),
        ]), max_steps=6)
        parent.add_tool(sous.as_tool(name="expert", description="délègue"))
        usage = parent.run("vas-y").usage
        # parent 320/25 + délégations 1200/600 — et pas 2200/1100
        assert usage.input_tokens == 1520
        assert usage.output_tokens == 625


class TestDelegationImbriquee:
    def test_le_petit_fils_remonte_jusqu_a_la_racine(self) -> None:
        """Le mécanisme est récursif : l'enfant absorbe déjà son propre enfant,
        donc la racine reçoit un total complet sans traitement spécial."""
        petit_fils = Agent(Fournisseur([_reponse(50, 25)]))
        enfant = Agent(Fournisseur([
            _reponse(10, 5, appel="petit"), _reponse(10, 5),
        ]), max_steps=4)
        enfant.add_tool(petit_fils.as_tool(name="petit", description="délègue encore"))

        racine = _superviseur(enfant)
        usage = racine.run("vas-y").usage
        # racine 220/15 + enfant (20/10) + petit-fils (50/25)
        assert usage.input_tokens == 290
        assert usage.output_tokens == 50


class TestEnParallele:
    def test_deux_specialistes_concurrents_sont_tous_deux_comptes(self) -> None:
        """En parallèle, la collecte se fait dans les threads mais l'ADDITION a
        lieu dans la boucle ordonnée — sinon des jetons se perdraient."""
        a = Agent(Fournisseur([_reponse(1000, 100)]))
        b = Agent(Fournisseur([_reponse(2000, 200)]))
        parent = Agent(Fournisseur([
            LLMResponse(tool_calls=[
                ToolCall(id="c1", name="expert_a", arguments={"request": "x"}),
                ToolCall(id="c2", name="expert_b", arguments={"request": "y"}),
            ], usage=TokenUsage(input_tokens=100, output_tokens=10)),
            _reponse(120, 5),
        ]), max_steps=4, parallel_tool_calls=True)
        parent.add_tool(a.as_tool(name="expert_a", description="A"))
        parent.add_tool(b.as_tool(name="expert_b", description="B"))
        usage = parent.run("vas-y").usage
        assert usage.input_tokens == 220 + 1000 + 2000
        assert usage.output_tokens == 15 + 100 + 200
