"""La porte de décision unique : tout ce qui AGIT passe par la même décision.

Avant, la boucle décidait pour un appel d'outil DIRECT (`TurnGuards` : anti-boucle,
garde trifecta, politique de l'hôte). Or le modèle agit aussi par d'autres chemins, que
cette décision ne voyait pas :

* le **code qu'il écrit** (`run_python`, un outil généré) appelle des fonctions de
  l'hôte par le pont `context["call_host"]` — sans politique, sans garde trifecta,
  sans teinte. L'éditeur d'un SDK d'agents le dit lui-même d'un mécanisme voisin :
  `allowed_callers` n'est pas une frontière de sécurité (Anthropic, documentation
  du « programmatic tool calling ») ;
* le dispatcher `call_host_function` d'`EvolutionRuntime`, et les outils promus en natif ;
* un **sous-agent** (`as_tool`, `delegate_to`) agit avec SA politique : celle du parent
  ne le couvre pas.

Ce module est LA décision, au singulier. `ActionGate.decide()` applique, dans cet
ordre, la garde trifecta puis la politique de l'hôte, sur un appel quelconque, et
rend `None` (autorisé) ou la raison du refus. Deux garanties de structure, vérifiées
par `tests/test_gate.py` :

* `agent.tool_policy` n'est appelé qu'à UN endroit du paquet (`evaluate_policy`) ;
* chaque fonction qui exécute une fonction de l'hôte consulte la porte AVANT.

Le contrat reste celui de la bibliothèque : une politique qui plante REFUSE
(fail-closed). Une approbation humaine ne peut pas faire de pause au milieu d'un
programme ni d'un sous-agent — l'appel serait déjà parti pour de bon — donc elle
REFUSE aussi, avec une raison lisible. Rien ne change pour un agent sans politique et
sans outil marqué `egress` : la porte laisse tout passer.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from .errors import ApprovalRequired
from .logging import get_logger
from .schema import Message, ToolCall, ToolSpec, is_tainted

if TYPE_CHECKING:
    from .agent import Agent, ToolPolicyContext
    from .guards import TurnGuards

__all__ = ["ActionGate", "active_gate", "evaluate_policy", "gate_scope", "trifecta_blocks"]

_log = get_logger("gate")

# La porte du run EN COURS pour le thread qui exécute un outil. Posée par la boucle
# autour de chaque exécution ; lue par les trois endroits qui lancent une fonction
# de l'hôte pour le compte d'un modèle. Hors d'un run d'agent (un `PythonRunner`
# appelé à la main, les self-tests du constructeur) : `None`, donc rien ne change.
_ACTIVE: contextvars.ContextVar[ActionGate | None] = contextvars.ContextVar(
    "autoagent_action_gate", default=None)


def active_gate() -> ActionGate | None:
    """La porte posée autour de l'outil en cours d'exécution, ou `None`."""
    return _ACTIVE.get()


@contextmanager
def gate_scope(gate: ActionGate) -> Iterator[ActionGate]:
    """Pose `gate` pour la durée de l'exécution d'un outil (imbriquable : un sous-agent
    qui exécute ses propres outils pose la SIENNE puis restaure celle du parent)."""
    token = _ACTIVE.set(gate)
    try:
        yield gate
    finally:
        _ACTIVE.reset(token)


def trifecta_blocks(agent: Agent, tainted: bool, egress: bool) -> bool:
    """La règle de la « lethal trifecta » : un outil `egress` est refusé dès que le run a
    ingéré du contenu non fiable (sauf `trifecta_guard="off"`). UNE définition, partagée
    par la boucle (`TurnGuards.trifecta`) et par la porte."""
    return agent.trifecta_guard != "off" and tainted and egress


def evaluate_policy(
    agent: Agent, ctx: ToolPolicyContext
) -> tuple[str | None, ApprovalRequired | None]:
    """L'UNIQUE endroit où le callable `tool_policy` de l'hôte est invoqué.

    Rend `(verdict, pause)` : `verdict` est `None` (autorisé) ou la raison d'un refus ;
    `pause` est l'`ApprovalRequired` levé par la politique, que l'appelant décide de
    propager (la boucle, avec un instantané) ou de convertir en refus (la porte, qui
    ne peut pas mettre un programme en pause). Fail-CLOSED : une politique qui plante,
    ou qui rend autre chose qu'un `str`/`None`, REFUSE.
    """
    try:
        verdict = agent.tool_policy(ctx) if agent.tool_policy is not None else None
    except ApprovalRequired as pause:
        return None, pause
    except Exception as exc:
        _log.exception("tool_policy raised; denying %r (fail-closed)", ctx.call.name)
        return f"policy error: {type(exc).__name__}: {exc}", None
    if verdict is not None and not isinstance(verdict, str):
        return "policy returned an unsupported verdict type", None
    return verdict, None


class ActionGate:
    """La décision pour les actions qu'un outil déclenche lui-même.

    Une porte par EXÉCUTION d'outil (créée par la boucle, voir `Agent._run_loop`). Elle
    connaît l'appel en cours, l'étape, le contexte de l'hôte et l'état de teinte du run ;
    `decide()` rend le verdict pour une action imbriquée : une fonction de l'hôte appelée
    depuis du code du modèle, ou un outil appelé par un sous-agent.

    `tainted_by_host` : une fonction de l'hôte marquée `untrusted` a rendu du contenu non
    fiable PENDANT le programme. La suite du même programme — et le run, une fois l'outil
    terminé — en sont teintés : sans cela, un `run_python` qui lit une page puis envoie
    un mail aurait la teinte « propre » au moment de l'envoi.
    """

    def __init__(self, agent: Agent, guards: TurnGuards, call: ToolCall, step: int,
                 context: dict[str, Any] | None) -> None:
        self.agent = agent
        self.guards = guards
        self.call = call
        self.step = step
        self.context = context
        self.tainted_by_host = False
        self._n = 0

    @property
    def tainted(self) -> bool:
        """Le run a-t-il ingéré du contenu non fiable (transcript, drapeau monotone, ou
        fonction de l'hôte non fiable appelée dans CE programme) ?"""
        return bool(self.guards.taint[0] or is_tainted(self.guards.messages) or self.tainted_by_host)

    def decide(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        spec: ToolSpec | None = None,
        source: str = "host_function",
        messages: list[Message] | None = None,
        step: int | None = None,
        own_tainted: bool = False,
    ) -> str | None:
        """Verdict pour UNE action : `None` = autorisée, sinon la raison du refus.

        `source` dit d'où vient l'action (`"host_function"` : appelée depuis du code du
        modèle ; `"subagent"` : appelée par un sous-agent) — la politique le lit dans
        `ctx.source`. `spec` porte les drapeaux `egress` / `untrusted` / permissions de la
        fonction ou de l'outil visé ; `own_tainted` ajoute la teinte propre d'un
        sous-agent à celle du run parent. Ne lève JAMAIS : une erreur ici est un refus.
        """
        try:
            return self._decide(name, arguments, spec, source, messages, step, own_tainted)
        except Exception as exc:                          # fail-CLOSED
            _log.exception("gate raised; denying %r", name)
            return f"gate error: {type(exc).__name__}: {exc}"

    def _decide(self, name: str, arguments: dict[str, Any] | None, spec: ToolSpec | None,
                source: str, messages: list[Message] | None, step: int | None,
                own_tainted: bool) -> str | None:
        from .agent import ToolPolicyContext  # import paresseux : évite le cycle

        agent = self.agent
        if source == "host_function" and not agent.govern_host_calls:
            return None                                    # l'option de retour au comportement 0.22.0
        self._n += 1
        call = ToolCall(id=f"{self.call.id}#{source}{self._n}", name=name,
                        arguments=dict(arguments or {}))
        egress = bool(spec is not None and spec.egress)
        tainted = self.tainted or own_tainted
        etape = self.step if step is None else step
        reason: str | None = None
        observe = False

        if trifecta_blocks(agent, tainted, egress):
            message = (
                f"EgressBlocked: `{name}` can send data out of the system and this run has "
                f"already ingested untrusted external content. The call was refused by "
                f"policy, not by the model."
            )
            if agent.trifecta_guard == "approve":
                message += " (An approval cannot pause a program or a delegate: it is refused.)"
            if agent.shadow_guards:
                observe = True                              # mode témoin : tracé, pas appliqué
            else:
                reason = message

        if reason is None and agent.tool_policy is not None:
            ctx = ToolPolicyContext(
                call=call, spec=spec, step=etape,
                messages=messages if messages is not None else self.guards.messages,
                context=self.context or {}, tainted=tainted, egress=egress, source=source,
            )
            verdict, pause = evaluate_policy(agent, ctx)
            if pause is not None:
                reason = (f"ApprovalRequired: {pause} — a human approval cannot pause a "
                          f"program or a delegate mid-run, so the call is refused")
            else:
                reason = verdict

        # L'héritage est TRANSITIF : un sous-agent qui hérite d'un parent, lui-même sous-agent
        # d'un autre, est borné par toute la chaîne — chaque porte consulte celle du dessus.
        parent = agent._parent_gate
        if reason is None and parent is not None:
            reason = parent.decide(name, arguments, spec=spec, source=source, messages=messages,
                                   step=step, own_tainted=tainted)

        if reason is None and spec is not None and spec.untrusted:
            self.tainted_by_host = True

        agent._emit("gate_decision", {
            "source": source, "name": name, "allowed": reason is None,
            "reason": reason if reason is None else reason[:300],
            "would_block": observe, "parent_call_id": self.call.id, "step": etape,
        })
        return reason
