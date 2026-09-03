"""Les gardes d'un tour — sorties de la boucle pour qu'on puisse LIRE les deux (0.21.0).

`agent.py` avait dépassé 2 300 lignes, et plusieurs centaines d'entre elles
étaient trois fermetures imbriquées dans le générateur de la boucle : la garde
anti-boucle, la garde trifecta, la politique de l'hôte. La thèse n°1 de la lib
est « une boucle, un fichier qu'on lit de bout en bout » ; elle ne tenait plus.

Ce module contient EXACTEMENT ces gardes, au comportement près : mêmes
événements de trace, mêmes charges utiles, même ordre — les fixtures de rejeu
(§23) et les 900 tests ne voient aucune différence. La boucle les appelle via
`TurnGuards`, construit une fois par run avec l'état qu'elles lisaient déjà
en fermeture : le transcript, la teinte, le compteur du mode témoin, le
snapshot pour une pause d'approbation.

Un changement de fond en profite : la garde anti-boucle ne RECOMPTE plus toutes
les signatures du transcript à chaque tour (80 600 appels pour un run de 400
étapes — quadratique). Elle tient un compteur INCRÉMENTAL sur les messages
qu'elle n'a pas encore vus. Même résultat, coût linéaire.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from .errors import ApprovalRequired
from .logging import get_logger
from .registry import ToolResult
from .schema import Message, ToolCall, is_tainted
from .trace import truncate_preview

if TYPE_CHECKING:
    from .agent import Agent, RunState

__all__ = ["TurnGuards", "_call_signature", "_count_call_signatures"]

_log = get_logger("guards")


def _call_signature(call: ToolCall) -> str:
    """Identity of a tool call for repetition detection: name + sorted arguments.

    Sorted keys so the same call written in a different key order counts as the
    same call. A plain string (not a hash) keeps traces debuggable.
    """
    try:
        args = json.dumps(call.arguments or {}, sort_keys=True, ensure_ascii=False, default=repr)
    except (TypeError, ValueError):  # pragma: no cover — arguments are JSON from the wire
        args = repr(call.arguments)
    return f"{call.name}({args})"


def _count_call_signatures(messages: Sequence[Message]) -> dict[str, int]:
    """How many times each call signature was already REQUESTED in this run."""
    counts: dict[str, int] = {}
    for message in messages:
        for call in message.tool_calls or ():
            signature = _call_signature(call)
            counts[signature] = counts.get(signature, 0) + 1
    return counts


class TurnGuards:
    """Les trois gardes d'un tour, dans l'ordre où la boucle les applique.

    `builtin()` = anti-boucle + trifecta, avec le mode témoin ; `policy()` = la
    politique de l'hôte, TOUJOURS appliquée (jamais observée), en dernier : elle
    ne peut qu'ajouter des refus, jamais lever ceux des gardes intégrées.
    """

    def __init__(
        self,
        agent: Agent,
        messages: list[Message],
        context: dict[str, Any] | None,
        taint: list[bool],
        temoin: list[int],
        snapshot: Callable[[int], RunState],
    ) -> None:
        self.agent = agent
        self.messages = messages              # le MÊME objet que la boucle fait grandir
        self.context = context
        self.taint = taint
        self.temoin = temoin
        self.snapshot = snapshot
        # Compteur incrémental de la garde anti-boucle : signatures des appels
        # déjà présents dans le transcript, mis à jour sur les seuls messages
        # pas encore vus. Seedé au premier usage (couvre un historique persisté
        # et la reprise).
        self._counts: dict[str, int] = {}
        self._seen_len = 0
        # Appels du tour COURANT déjà autorisés en avance (§33) mais pas encore
        # dans le transcript : la vérification réelle du tour les verra tous
        # dans le même message assistant, la vérification anticipée doit donc
        # les compter aussi — sinon deux appels identiques d'un même tour
        # partiraient tous deux en avance pour être refusés ensuite.
        self._pending: dict[str, int] = {}

    # ── anti-boucle ──────────────────────────────────────────────────────────

    def _observe_transcript(self) -> None:
        """Ajoute au compteur les appels des messages pas encore vus — même
        règle de comptage que `_count_call_signatures` sur le transcript entier,
        mais en O(nouveaux messages) au lieu de O(transcript) par tour."""
        if len(self.messages) > self._seen_len:
            self._pending.clear()                  # le tour est entré au transcript
        for m in self.messages[self._seen_len:]:
            for call in m.tool_calls or ():
                sig = _call_signature(call)
                self._counts[sig] = self._counts.get(sig, 0) + 1
        self._seen_len = len(self.messages)

    def loop_guard(
        self, calls: list[ToolCall], step: int, req_span: str | None, *, pending: bool = False
    ) -> dict[str, ToolResult]:
        """Refuse a tool call the model has already made IDENTICALLY N times.

        An agent that re-issues the same (name, arguments) call in a loop
        burns `max_steps` and the whole `token_budget` at full price, and
        re-runs side effects each time. `max_steps` bounds the damage but
        does not diagnose it. Here the repetition is CODE-detected and the
        model gets a deterministic refusal on the SAME channel a policy
        denial uses — the re-planning path that already works — instead of
        a prompt begging it to stop.

        Counted from the transcript (like taint and revealed tools), so the
        count survives checkpoint/resume with no extra RunState field.

        `pending=True` : l'appel n'est PAS encore dans le transcript (vérification
        anticipée pendant le flux, §33). On le compte quand même, pour que la
        vérification anticipée et la vérification réelle donnent le MÊME verdict —
        avant 0.21.0 l'anticipée était plus permissive d'une unité, et un outil
        pouvait partir en avance pour être refusé ensuite.
        """
        agent = self.agent
        overrides: dict[str, ToolResult] = {}
        if agent.max_repeated_tool_calls is None:
            return overrides
        self._observe_transcript()
        # NB: l'appel COURANT est déjà dans le transcript (le message assistant
        # est ajouté avant l'exécution) — `seen` compte donc les demandes
        # jusqu'ici INCLUSE. On bloque au-delà du plafond, pas à l'atteinte,
        # sinon `max_repeated_tool_calls=1` refuserait le PREMIER appel.
        for call in calls:
            signature = _call_signature(call)
            seen = self._counts.get(signature, 0) + self._pending.get(signature, 0) + (1 if pending else 0)
            if seen <= agent.max_repeated_tool_calls:
                if pending:
                    self._pending[signature] = self._pending.get(signature, 0) + 1
                continue
            overrides[call.id] = ToolResult(
                ok=False,
                error=(
                    f"RepeatedCall: you already called `{call.name}` with these exact "
                    f"arguments {agent.max_repeated_tool_calls} times and it was not run "
                    f"again. Change the arguments, use a different tool, or stop and "
                    f"report what you have."
                ),
            )
            agent._emit(
                "loop_guard_would_block" if agent.shadow_guards else "loop_guard_block",
                {"step": step, "name": call.name, "call_id": call.id, "repeats": seen},
                parent_id=req_span,
            )
        return overrides

    # ── trifecta ─────────────────────────────────────────────────────────────

    def trifecta(self, calls: list[ToolCall], step: int, req_span: str | None) -> dict[str, ToolResult]:
        """Block an EGRESS tool once the run has ingested untrusted content.

        The lethal trifecta made concrete: private data + untrusted content +
        a way out = exfiltration by indirect injection, with no software
        vulnerability involved. The library already instrumented the first two
        legs (`untrusted=True`, network-less sandbox); this closes the third
        so a host that forgets the rule is not exfiltrable by default.

        Fires only for tools the host explicitly marked ``egress=True`` — no
        existing code has that flag, so the default ``"deny"`` cannot change
        the behaviour of an already-deployed agent.
        """
        agent = self.agent
        overrides: dict[str, ToolResult] = {}
        if agent.trifecta_guard == "off":
            return overrides
        if not (self.taint[0] or is_tainted(self.messages)):
            return overrides                      # rien d'externe n'est entré
        for call in calls:
            if not agent._is_egress(call):
                continue
            if agent.trifecta_guard == "approve":
                pause = ApprovalRequired(
                    f"egress tool `{call.name}` requested after untrusted content "
                    f"entered the run (lethal trifecta) — human approval required"
                )
                pause.state = self.snapshot(step)     # aucun outil du tour n'a tourné
                pause.calls = list(calls)
                agent._emit(
                    "trifecta_approval_required",
                    {"step": step, "name": call.name, "call_id": call.id},
                    parent_id=req_span,
                )
                raise pause
            overrides[call.id] = ToolResult(
                ok=False,
                error=(
                    f"EgressBlocked: `{call.name}` can send data out of the system and "
                    f"this run has already ingested untrusted external content. The "
                    f"call was refused by policy, not by the model. Summarise your "
                    f"finding to the user instead of transmitting it."
                ),
            )
            agent._emit(
                "trifecta_would_block" if agent.shadow_guards else "trifecta_block",
                {"step": step, "name": call.name, "call_id": call.id},
                parent_id=req_span,
            )
        return overrides

    # ── politique de l'hôte ──────────────────────────────────────────────────

    def policy(self, calls: list[ToolCall], step: int, req_span: str | None) -> dict[str, ToolResult]:
        """Consult tool_policy for the WHOLE turn before any side effect.

        Returns {call_id: denial ToolResult} for denied calls. Raises
        ApprovalRequired (with a resumable snapshot attached) BEFORE
        anything of the turn has executed — a pause must never land
        after a side effect.
        """
        from .agent import ToolPolicyContext  # import paresseux : évite le cycle

        agent = self.agent
        overrides: dict[str, ToolResult] = {}
        if agent.tool_policy is None:
            return overrides
        tainted = self.taint[0] or is_tainted(self.messages)  # état AVANT le tour
        for call in calls:
            spec = next((s for s in agent.registry.specs() if s.name == call.name), None)
            policy_ctx = ToolPolicyContext(
                call=call, spec=spec, step=step,
                messages=self.messages, context=self.context or {},
                tainted=tainted,
                egress=bool(spec is not None and spec.egress),
            )
            try:
                verdict = agent.tool_policy(policy_ctx)
            except ApprovalRequired as pause:
                pause.state = self.snapshot(step)  # LLM call done, zero tools executed
                pause.calls = list(calls)
                agent._emit(
                    "approval_required",
                    {
                        "step": step,
                        "call_id": call.id,
                        "names": [c.name for c in calls],
                        "reason": truncate_preview(str(pause)),
                    },
                    parent_id=req_span,
                )
                raise
            except Exception as exc:
                # Fail-CLOSED: a buggy policy denies. This hook is a
                # security boundary — the opposite contract of trace/
                # checkpoint callbacks, which fail-open.
                _log.exception("tool_policy raised; denying %r (fail-closed)", call.name)
                verdict = f"policy error: {type(exc).__name__}: {exc}"
            if verdict is None:
                continue
            if not isinstance(verdict, str):
                verdict = "policy returned an unsupported verdict type"
            overrides[call.id] = ToolResult(ok=False, error=f"ToolPolicyDenied: {verdict}")
            agent._emit(
                "tool_policy_deny",
                {"name": call.name, "call_id": call.id, "step": step,
                 "reason": truncate_preview(verdict)},
                parent_id=req_span,
            )
        return overrides

    # ── composition ──────────────────────────────────────────────────────────

    def builtin(
        self, calls: list[ToolCall], step: int, req_span: str | None, *, pending: bool = False
    ) -> dict[str, ToolResult]:
        """Anti-boucle + trifecta, avec le MODE TÉMOIN : le verdict est calculé et
        tracé (`*_would_block`) mais pas appliqué — le radar photographie sans
        verbaliser (§30). Le compteur du témoin est incrémenté ici."""
        integres = self.loop_guard(calls, step, req_span, pending=pending)
        integres.update(self.trifecta(calls, step, req_span))
        if self.agent.shadow_guards and integres:
            self.temoin[0] += len(integres)
            return {}
        return integres
