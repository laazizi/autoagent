"""Cascade par RÉSULTAT : le modèle bon marché d'abord, le gros seulement si TON juge dit non (0.21.0).

    palier 1 (pas cher)  →  check(résultat) ?  oui → fini
                                 │ non
    palier 2 (plus cher) →  check(résultat) ?  oui → fini
                                 │ non
    …                           → on rend le dernier, et on dit qu'il a échoué

`RoutingProvider` (§9) route AVANT l'appel, sur la forme de la requête (une
image → le modèle vision). Ici on route APRÈS, sur le résultat : ce que le petit
modèle a produit est-il acceptable ? Les cascades (FrugalGPT, LLM-Cascade)
tiennent leur économie de là — la plupart des tâches n'ont pas besoin du gros
modèle, mais on ne le sait qu'après.

CE QUI DÉCIDE DE MONTER EST DU CODE, PAS LE MODÈLE. Si c'est le petit modèle qui
dit « je ne suis pas sûr », on est revenu à une consigne qu'on espère respectée.
`check` est le juge de l'HÔTE — le même contrat que `run_k` (§25.4) : un
prédicat déterministe sur `AgentResult`. Un `check` qui lève compte comme un
refus, jamais comme une acceptation.

Ce que la cascade GARANTIT :

  * les paliers ratés SE PAIENT — `CascadeResult.usage` cumule tout, y compris
    les essais rejetés ; c'est la vraie facture de la cascade, pas celle du seul
    palier qui a répondu ;
  * une PAUSE D'APPROBATION n'est pas un échec : `ApprovalRequired` (et
    `AgentCancelled`) REMONTENT tels quels. Monter de palier sur une pause
    reviendrait à contourner le feu vert humain avec un autre modèle — la
    cascade ne doit jamais devenir une porte dérobée dans `tool_policy` ;
  * un palier qui plante (`MaxStepsExceeded`, `TokenBudgetExceeded`, erreur
    fournisseur…) est un palier RATÉ, et la cascade continue.

Ce que la cascade ne fait pas : deviner. Sans `check` déterministe, il n'y a pas
de cascade — il y a un pari.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .errors import AgentCancelled, ApprovalRequired
from .logging import get_logger
from .schema import TokenUsage

if TYPE_CHECKING:
    from .agent import Agent, AgentResult

__all__ = ["CascadeResult", "TierAttempt", "cascade"]

_log = get_logger("cascade")


@dataclass
class TierAttempt:
    """Un palier essayé : quel modèle, a-t-il convaincu le juge, à quel prix."""

    index: int
    model: str | None
    ok: bool = False
    steps: int = 0
    usage: TokenUsage | None = None
    error: str = ""


@dataclass
class CascadeResult:
    result: Any                                   # AgentResult du palier accepté, ou du dernier
    tier: int | None                              # palier (1-based) qui a convaincu, None si aucun
    attempts: list[TierAttempt] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        return self.tier is not None

    @property
    def escalations(self) -> int:
        """Combien de fois on est monté d'un palier (0 = le premier a suffi)."""
        return max(0, len(self.attempts) - 1)

    @property
    def usage(self) -> TokenUsage | None:
        """La dépense de TOUS les paliers, ratés compris — la vraie facture."""
        vus = [a.usage for a in self.attempts if a.usage is not None]
        if not vus:
            return None
        caches = [u.cached_tokens for u in vus if u.cached_tokens is not None]
        return TokenUsage(
            input_tokens=sum(u.input_tokens or 0 for u in vus),
            output_tokens=sum(u.output_tokens or 0 for u in vus),
            total_tokens=sum(u.total_tokens or 0 for u in vus),
            cached_tokens=sum(caches) if caches else None,
        )

    def summary(self) -> str:
        chemin = " → ".join(f"{a.model or '?'}{'✓' if a.ok else '✗'}" for a in self.attempts)
        prix = f" · {self.usage.total_tokens} jetons" if self.usage and self.usage.total_tokens else ""
        if self.accepted:
            return f"accepté au palier {self.tier} : {chemin}{prix}"
        return f"aucun palier accepté : {chemin}{prix}"


def _model_of(agent: Any) -> str | None:
    config = getattr(getattr(agent, "provider", None), "config", None)
    return getattr(config, "model", None)


def cascade(
    tiers: Sequence[Agent | Callable[[], Agent]],
    prompt: str,
    *,
    check: Callable[[AgentResult], bool],
    context: dict[str, Any] | None = None,
    on_tier: Callable[[TierAttempt], None] | None = None,
) -> CascadeResult:
    """Essaie les paliers dans l'ordre ; s'arrête au premier que `check` accepte.

    Args:
        tiers: agents du moins cher au plus cher — ou des fabriques sans argument
            (préférer les fabriques dès que les outils ont des effets de bord :
            chaque palier doit partir d'un état propre).
        prompt: la tâche, identique à chaque palier.
        check: le juge de l'HÔTE, `(AgentResult) -> bool`, déterministe. S'il
            lève, le palier est refusé et le message atterrit dans
            `TierAttempt.error` — on n'avale pas un juge cassé.
        context: passé tel quel à `agent.run(context=...)`.
        on_tier: rappel après chaque palier (journalisation, progression).

    Rend un `CascadeResult` : le résultat accepté (ou le dernier obtenu si aucun
    ne l'a été), le palier qui a répondu, et la dépense CUMULÉE.

    `ApprovalRequired` et `AgentCancelled` remontent immédiatement : une pause
    humaine n'est pas un échec qu'un modèle plus gros pourrait « rattraper ».
    """
    if not tiers:
        raise ValueError("cascade attend au moins un palier")
    if not callable(check):
        raise TypeError("check must be a callable (AgentResult) -> bool")

    resultat = CascadeResult(result=None, tier=None)
    dernier: Any = None
    for index, palier in enumerate(tiers, 1):
        agent = palier() if callable(palier) and not hasattr(palier, "run") else palier
        essai = TierAttempt(index=index, model=_model_of(agent))
        try:
            dernier = agent.run(prompt, context=context)
            essai.steps = getattr(dernier, "steps", 0)
            essai.usage = getattr(dernier, "usage", None)
            try:
                essai.ok = bool(check(dernier))
            except Exception as exc:
                essai.error = f"check raised: {type(exc).__name__}: {exc}"
        except (ApprovalRequired, AgentCancelled):
            raise                                 # une pause n'est pas un échec
        except Exception as exc:
            # Un palier qui plante EST un palier raté ; la dépense engagée est
            # portée par l'exception quand la boucle la connaît (state).
            essai.error = f"{type(exc).__name__}: {exc}"
            etat = getattr(exc, "state", None)
            if etat is not None:
                essai.usage = TokenUsage(
                    input_tokens=getattr(etat, "input_tokens", 0),
                    output_tokens=getattr(etat, "output_tokens", 0),
                )
                essai.usage.total_tokens = (essai.usage.input_tokens or 0) + (essai.usage.output_tokens or 0)
        resultat.attempts.append(essai)
        if on_tier is not None:
            try:
                on_tier(essai)
            except Exception:
                _log.exception("on_tier callback failed")   # fail-open
        if essai.ok:
            resultat.result = dernier
            resultat.tier = index
            return resultat
    resultat.result = dernier
    return resultat
