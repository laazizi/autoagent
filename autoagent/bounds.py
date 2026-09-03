"""`Bounds` — les bornes d'un agent en UN objet, lisible d'un coup (0.21.0).

`Agent.__init__` avait vingt paramètres, dont huit sont des BORNES : le plafond
d'étapes, le plafond de jetons, la troncature d'un résultat, l'élagage des
vieux résultats et son lot, la garde anti-boucle, la garde trifecta, le mode
témoin. Éparpillées dans une signature, elles se lisent mal, se partagent mal
entre agents, et ne se sérialisent pas dans une trace. La thèse de la lib est
« les bornes sont du code » — elles méritent d'être un objet.

    from autoagent import Agent, Bounds

    PROD = Bounds(max_steps=12, token_budget=40_000, max_tool_result_chars=4_000,
                  prune_tool_results_after=2, max_repeated_tool_calls=3)

    agent = Agent(provider, bounds=PROD)          # les huit d'un coup
    agent.bounds                                  # relecture : ce qui est EN VIGUEUR
    agent.bounds.to_dict()                        # dans une trace, un rapport, un JSON

Rétrocompatible par construction : les vingt kwargs restent, `bounds` s'ajoute.
Quand les deux sont donnés, **un kwarg explicite l'emporte** sur le champ
correspondant de `bounds` — l'intention la plus locale gagne. Les attributs
`agent.max_steps`, `agent.token_budget`… restent des attributs ordinaires,
lisibles et modifiables (la démo 22 relève `agent.token_budget` avant un
`resume` : ça continue de marcher) ; `agent.bounds` en est une PHOTO au moment
où on la demande, pas une copie figée à la construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import Agent

__all__ = ["Bounds"]


@dataclass(frozen=True)
class Bounds:
    """Les huit bornes d'un agent. Chaque champ a la même sémantique et la même
    valeur par défaut que le kwarg `Agent(...)` du même nom.

    Attributes:
        max_steps: plafond d'itérations LLM↔outils d'un run.
        token_budget: plafond dur de jetons cumulés (None = illimité).
        max_tool_result_chars: troncature MILIEU d'un résultat d'outil (§24.1).
        prune_tool_results_after: au-delà des N derniers, un résultat perd sa
            charge dans la vue envoyée au fournisseur (§28).
        prune_batch: élaguer par lots de K pour ne pas casser le cache (§28.4).
        max_repeated_tool_calls: refuser le (N+1)ᵉ appel identique (§24.2).
        trifecta_guard: "deny" | "approve" | "off" (§24.3).
        shadow_guards: observer les gardes sans les appliquer (§30).
    """

    max_steps: int = 8
    token_budget: int | None = None
    max_tool_result_chars: int | None = None
    prune_tool_results_after: int | None = None
    prune_batch: int = 1
    max_repeated_tool_calls: int | None = None
    trifecta_guard: str = "deny"
    shadow_guards: bool = False

    @classmethod
    def from_agent(cls, agent: Agent) -> Bounds:
        """La photo des bornes EN VIGUEUR sur un agent (ce que la boucle lira)."""
        return cls(**{f.name: getattr(agent, f.name) for f in fields(cls)})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def apply_to(self, agent: Agent, explicit: dict[str, Any]) -> None:
        """Pose les bornes sur `agent`, sauf celles que l'appelant a données
        EXPLICITEMENT en kwarg (`explicit` : nom → valeur non-défaut)."""
        for f in fields(self):
            if f.name not in explicit:
                setattr(agent, f.name, getattr(self, f.name))

    @staticmethod
    def names() -> tuple[str, ...]:
        return tuple(f.name for f in fields(Bounds))
