"""`shadow_guards` — observer une garde au lieu de la subir (0.20.0).

Activer une borne était un pari : impossible de savoir si elle refusera quelque
chose de légitime avant qu'elle ne le fasse, en production. Résultat courant :
on ne l'active pas, ou on l'active une fois, ça casse, et on l'éteint pour
toujours.

En mode témoin, le verdict est calculé et TRACÉ (`*_would_block`) mais pas
appliqué. Le radar photographie sans verbaliser. Une semaine plus tard on lit
« cette borne aurait refusé N appels, les voici » et on décide en sachant.

Ce que ces tests verrouillent :

  1. RIEN N'EST REFUSÉ en mode témoin — sinon ce n'est pas un mode témoin.
  2. LA TRACE DIT LEQUEL DES DEUX : `*_would_block` et `*_block` ne doivent
     jamais se confondre, sinon le rapport de fin de semaine est un mensonge.
  3. `tool_policy` N'EST JAMAIS OBSERVÉ. C'est la frontière de l'HÔTE : un
     drapeau de bibliothèque ne doit pas pouvoir éteindre le code qu'un hôte a
     écrit pour dire non.
"""

from __future__ import annotations

from autoagent import Agent, TraceEmitter
from autoagent.providers.base import LLMProvider
from autoagent.schema import LLMResponse, ModelConfig, TokenUsage, ToolCall


class Fournisseur(LLMProvider):
    def __init__(self, reponses: list[LLMResponse]) -> None:
        super().__init__(ModelConfig(provider="f", model="f", api_key="x"))
        self.reponses = list(reponses)

    def complete(self, request):  # type: ignore[no-untyped-def]
        return self.reponses.pop(0) if self.reponses else LLMResponse(content="fini")


def _repete(nom: str, n: int) -> list[LLMResponse]:
    return [LLMResponse(tool_calls=[ToolCall(id=f"c{i}", name=nom, arguments={})],
                        usage=TokenUsage(input_tokens=10, output_tokens=1))
            for i in range(n)] + [LLMResponse(content="fini")]


class Bilan:
    def __init__(self) -> None:
        self.types: list[str] = []
        self.fin: dict = {}

    def __call__(self, ev) -> None:  # type: ignore[no-untyped-def]
        self.types.append(ev.type)
        if ev.type == "run_end":
            self.fin = dict(ev.payload)


def _run_boucle(*, temoin: bool, tours: int = 6) -> tuple[int, Bilan]:
    bilan = Bilan()
    executions = {"n": 0}
    with TraceEmitter(on_event=bilan) as trace:
        agent = Agent(Fournisseur(_repete("lire", tours)), max_steps=tours + 2,
                      trace=trace, max_repeated_tool_calls=2, shadow_guards=temoin)

        @agent.tool
        def lire() -> str:
            """Lit toujours la même chose."""
            executions["n"] += 1
            return "ok"

        agent.run("vas-y")
    return executions["n"], bilan


class TestGardeAntiBoucle:
    def test_mode_normal_le_refus_s_applique(self) -> None:
        executions, bilan = _run_boucle(temoin=False)
        assert executions == 2, "la garde n'a pas refusé le 3e appel identique"
        assert "loop_guard_block" in bilan.types
        assert "loop_guard_would_block" not in bilan.types

    def test_mode_temoin_rien_n_est_refuse(self) -> None:
        executions, bilan = _run_boucle(temoin=True)
        assert executions == 6, "le mode témoin a bloqué — ce n'est pas un témoin"
        assert "loop_guard_would_block" in bilan.types
        assert "loop_guard_block" not in bilan.types, (
            "un événement de refus RÉEL en mode témoin rendrait le rapport faux")

    def test_le_bilan_compte_ce_qui_aurait_ete_refuse(self) -> None:
        _, bilan = _run_boucle(temoin=True)
        assert bilan.fin["shadow_guards"] is True
        assert bilan.fin["would_block"] == 4, "6 appels, 2 autorisés, 4 observés"

    def test_hors_mode_temoin_la_cle_est_ABSENTE(self) -> None:
        """« Aucune garde n'aurait bloqué » et « le mode n'était pas actif » ne
        sont pas le même fait — même règle que pour `cached_tokens`."""
        _, bilan = _run_boucle(temoin=False)
        assert "would_block" not in bilan.fin
        assert "shadow_guards" not in bilan.fin


class TestTrifecta:
    def _run(self, temoin: bool) -> tuple[list[str], Bilan]:
        bilan = Bilan()
        faits: list[str] = []
        with TraceEmitter(on_event=bilan) as trace:
            agent = Agent(Fournisseur([
                LLMResponse(tool_calls=[ToolCall(id="t1", name="lire_page", arguments={})],
                            usage=TokenUsage(input_tokens=10, output_tokens=1)),
                LLMResponse(tool_calls=[ToolCall(id="t2", name="envoyer", arguments={})],
                            usage=TokenUsage(input_tokens=10, output_tokens=1)),
                LLMResponse(content="fini"),
            ]), max_steps=5, trace=trace, shadow_guards=temoin)

            @agent.tool(untrusted=True)
            def lire_page() -> str:
                """Lit une page web."""
                return "contenu externe"

            @agent.tool(egress=True)
            def envoyer() -> str:
                """Envoie un e-mail."""
                faits.append("envoi")
                return "envoyé"

            agent.run("vas-y")
        return faits, bilan

    def test_mode_normal_la_sortie_est_bloquee(self) -> None:
        faits, bilan = self._run(temoin=False)
        assert faits == [], "l'envoi a eu lieu alors que le run était teinté"
        assert "trifecta_block" in bilan.types

    def test_mode_temoin_la_sortie_passe_mais_est_signalee(self) -> None:
        faits, bilan = self._run(temoin=True)
        assert faits == ["envoi"]
        assert "trifecta_would_block" in bilan.types
        assert "trifecta_block" not in bilan.types


class TestLaFrontiereDeLHote:
    """Le test qui compte le plus : un drapeau de bibliothèque ne doit pas
    pouvoir éteindre le refus écrit par l'hôte."""

    def test_tool_policy_n_est_JAMAIS_observee(self) -> None:
        faits: list[str] = []

        def politique(ctx) -> str | None:  # type: ignore[no-untyped-def]
            return "interdit par l'hôte" if ctx.name == "ecrire" else None

        agent = Agent(Fournisseur([
            LLMResponse(tool_calls=[ToolCall(id="c1", name="ecrire", arguments={})],
                        usage=TokenUsage(input_tokens=10, output_tokens=1)),
            LLMResponse(content="fini"),
        ]), max_steps=4, tool_policy=politique, shadow_guards=True)

        @agent.tool
        def ecrire() -> str:
            """Écrit en base."""
            faits.append("ecriture")
            return "écrit"

        agent.run("vas-y")
        assert faits == [], (
            "shadow_guards a désactivé la politique de l'hôte — "
            "une bibliothèque ne doit pas pouvoir faire ça")


class TestDefaut:
    def test_eteint_par_defaut(self) -> None:
        agent = Agent(Fournisseur([]))
        assert agent.shadow_guards is False
