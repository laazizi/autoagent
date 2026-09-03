"""34 — L'efficacité lue dans la trace, après coup — sans rien ajouter au run.

Chaque run écrit déjà une trace JSONL (`TraceEmitter`, démo 04). Elle répond à
des questions que le taux de réussite ne pose pas :

    Combien d'appels d'outils étaient REDONDANTS (même outil, mêmes arguments,
    même run) ? Combien REFUSÉS, et par quelle garde ? Combien de jetons par
    SUCCÈS, échecs compris ? Combien lancés en avance, combien élagués ?

    summarize_trace("trace.jsonl")      # ou une liste d'événements

Le constat de Probe&Prefill (arXiv 2605.09252) — près de la moitié des appels
d'outils inutiles sur leur banc — n'est mesurable chez nous que comme ça : on
ne lit pas dans le modèle, on lit dans ce qu'il a fait.

Aucune clé API : la première partie lit la trace déjà dans le dépôt
(`trace_demo.jsonl`, écrite par la démo 04) ; la seconde rejoue un modèle
scripté qui boucle, pour montrer redondance, gardes et élagage dans la même
lecture.

    python examples_autoagent/34_metriques_de_trace.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

from autoagent import Agent, TraceEmitter, summarize_trace  # noqa: E402
from autoagent.providers.base import LLMProvider  # noqa: E402
from autoagent.schema import LLMResponse, ModelConfig, TokenUsage, ToolCall  # noqa: E402

ICI = Path(__file__).resolve().parent
TRACE_DEMO = ICI / "trace_demo.jsonl"


class ModeleQuiInsiste(LLMProvider):
    """Redemande 5 fois le MÊME appel, puis conclut — le défaut à mesurer."""

    def __init__(self) -> None:
        super().__init__(ModelConfig(provider="scripte", model="insistant", api_key="x"))
        self.restants = 5

    def complete(self, request):  # type: ignore[no-untyped-def]
        if self.restants <= 0:
            return LLMResponse(content="fini", usage=TokenUsage(input_tokens=200, output_tokens=10))
        self.restants -= 1
        return LLMResponse(
            tool_calls=[ToolCall(id=f"c{self.restants}", name="statut", arguments={"capteur": "A12"})],
            usage=TokenUsage(input_tokens=200, output_tokens=8))


def _afficher(titre: str, m) -> None:  # type: ignore[no-untyped-def]
    print(f"\n  {titre}")
    for ligne in m.summary().splitlines():
        print(f"    {ligne}")


def main() -> None:
    print("PARTIE 1 — la trace déjà dans le dépôt (écrite par la démo 04)")
    print("─" * 76)
    m1 = summarize_trace(TRACE_DEMO)
    _afficher(f"{TRACE_DEMO.name}", m1)
    print("\n  Deux runs, dont un arrêté par le plafond de jetons : il compte dans la")
    print("  dépense, pas dans les succès. C'est pour ça que « jetons/succès » est")
    print("  plus grand que le coût du seul run réussi — les échecs se paient aussi.")

    print("\n\nPARTIE 2 — un modèle qui insiste, lu à travers sa trace")
    print("─" * 76)
    evenements: list = []
    with TraceEmitter(on_event=evenements.append) as trace:
        agent = Agent(ModeleQuiInsiste(), max_steps=8, trace=trace,
                      max_repeated_tool_calls=2, prune_tool_results_after=1)

        @agent.tool
        def statut(capteur: str) -> str:
            """Interroge un capteur."""
            return "ok " * 400

        agent.run("Donne le statut du capteur A12.")
    m2 = summarize_trace(evenements)
    _afficher("run avec max_repeated_tool_calls=2 et prune_tool_results_after=1", m2)

    print("\n  Lecture : 5 appels demandés, 4 REDONDANTS (même outil, mêmes arguments).")
    print("  La garde anti-boucle en a refusé 3 — les deux premiers ont tourné, le")
    print("  reste a reçu un refus déterministe. L'élagage a retiré les vieux")
    print("  résultats de la vue. Tout ça sans avoir ajouté une ligne au run : la")
    print("  trace le savait déjà, personne ne le lisait.")

    print("\n" + "─" * 76)
    print("CE QU'IL FAUT RETENIR\n")
    print("  Ces métriques ne valent que sur du TRAFIC RÉEL : ici elles mesurent une")
    print("  démo. Branche `summarize_trace` sur les traces JSONL de production et tu")
    print("  sauras, sans rien changer aux agents, où partent les jetons :")
    print("  appels redondants, refus des gardes, coût par succès. Puis tu règles —")
    print("  et tu remesures. La trace est le compteur ; la lib ne fait que le lire.")


if __name__ == "__main__":
    main()
