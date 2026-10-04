"""Erreurs de la bibliothèque — et les ATTRIBUTS qu'elles portent, déclarés.

Jusqu'en 0.20, `exc.state`, `exc.spent`, `exc.messages`, `exc.calls` étaient
posés dynamiquement sur les exceptions par la boucle. Ça marchait, mais aucun
éditeur ne les complétait, aucun typage ne les vérifiait (27 erreurs mypy), et
seule la doc savait qu'ils existaient. Ils sont maintenant déclarés ici, avec
des valeurs par défaut : le constructeur reste `Exception(message)` — un hôte
qui les levait ou les attrapait ne voit rien changer — et la boucle continue de
les renseigner après construction. Découvrable, typé, rétrocompatible (0.21.0).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import RunState
    from .schema import Message, ToolCall

__all__ = [
    "AgentCancelled",
    "ApprovalRequired",
    "AutoAgentError",
    "JournalCorrupted",
    "JournalError",
    "JournalLocked",
    "MCPError",
    "MaxStepsExceeded",
    "OutcomeUnknown",
    "ProviderError",
    "ReplayMismatch",
    "TokenBudgetExceeded",
    "ToolError",
    "ToolValidationError",
]


class AutoAgentError(Exception):
    """Base error for autoagent."""


class _ResumableError(AutoAgentError):
    """Une erreur qui porte de quoi REPRENDRE le run (0.21.0).

    Attributes (renseignés par la boucle avant de lever) :
        state: snapshot `RunState` à donner à `Agent.resume`, ou None si la
            boucle n'a pas pu en produire (erreur avant le premier tour).
        messages: la conversation au moment de l'arrêt.
        step: l'étape à laquelle l'arrêt a eu lieu.
    """

    def __init__(self, message: str = "", *args: Any) -> None:
        super().__init__(message, *args)
        self.state: RunState | None = None
        self.messages: list[Message] = []
        self.step: int | None = None


class AgentCancelled(_ResumableError):
    """Raised when an agent run is cancelled cooperatively via `cancel_token`.

    The lib reads the token at four moments (0.23.1): at the start of every
    loop iteration (no new provider call), between the chunks of a stream, before
    each tool call that has not started (it does NOT run — its result says so),
    and at the step boundary. A sub-agent run through `as_tool` / `delegate_to`
    receives the same token. A provider call already sent (not streamed) and a
    tool already running are not interrupted — cancellation happens at the next
    safe boundary.

    Attributes: ``state`` (resumable snapshot), ``step``.
    """


class ProviderError(AutoAgentError):
    """Raised when an LLM provider request fails.

    Attributes:
        status_code: HTTP status when the failure came from an HTTP error
            response (``None`` for network-level failures — DNS, timeout,
            connection reset — and for non-HTTP failures like bad JSON).
        retryable: ``True`` when the failure class is worth retrying
            (429 / 5xx / transient network error). Hosts can branch on
            this instead of parsing the message text.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class ApprovalRequired(_ResumableError):
    """Raised BY a ``tool_policy`` hook to pause the run for human approval.

    The agent loop catches it before ANY tool of the turn has executed,
    attaches a resumable snapshot, and re-raises to the host.

    Attributes (attached by the loop):
        state: ``RunState`` snapshot — feed it to ``Agent.resume`` once
            the human has decided. On resume the pending tool calls go
            through the policy AGAIN: an unapproved call pauses again
            (idempotent), a rejected one should get a ``str`` verdict so
            the model sees the refusal and re-plans.
        calls: The turn's pending ``ToolCall`` list (nothing executed).
    """

    def __init__(self, message: str = "", *args: Any) -> None:
        super().__init__(message, *args)
        self.calls: list[ToolCall] = []


class ReplayMismatch(AutoAgentError):
    """Raised during replay when the run DIVERGES from the recorded fixture
    (0.16.0).

    Either the request signature at position N no longer matches the recorded
    one (different tool requested, different message shape — with ``strict``),
    or the run asks for more calls than the fixture holds. This divergence is
    a FEATURE: it tells you the agent's behaviour changed since the recording
    (a prompt edit, a code change), and points at the exact step.
    """


class MCPError(AutoAgentError):
    """Raised when an MCP server interaction fails.

    Covers transport failures (server not launchable, closed pipe,
    response timeout) and JSON-RPC error responses (the code and message
    are included in the text). A tool result flagged ``isError`` is NOT
    an ``MCPError`` — it raises ``ToolError`` so the registry surfaces
    it to the LLM as an ordinary tool error.
    """


class ToolError(AutoAgentError):
    """Raised when a tool cannot be executed."""


class ToolValidationError(ToolError):
    """Raised when generated tool code is rejected."""


class MaxStepsExceeded(_ResumableError):
    """Raised when the agent loop reaches its configured step limit.

    Attributes: ``state`` (resumable — raise ``max_steps`` and call
    ``Agent.resume``), ``messages``.
    """


class TokenBudgetExceeded(_ResumableError):
    """Raised when a run's cumulative token usage reaches ``token_budget``.

    Checked BEFORE each provider call (the call that crossed the budget is
    never truncated mid-flight). Only enforceable when the provider reports
    usage — unreported calls count as zero (best effort, never invented).

    Attributes: ``messages`` (the conversation so far), ``spent``
    (input+output tokens), ``state`` (resumable — raise the budget and call
    ``Agent.resume``).
    """

    def __init__(self, message: str = "", *args: Any) -> None:
        super().__init__(message, *args)
        self.spent: int = 0


class JournalError(AutoAgentError):
    """Le journal durable n'a pas pu enregistrer (disque plein, fichier verrouillé ou
    corrompu). FAIL-CLOSED : un effet dont l'INTENTION n'a pas pu être écrite ne part pas.
    C'est le contrat inverse de la trace (observabilité, fail-open) — un journal qui se tait
    sans rien dire n'est plus un journal."""


class JournalLocked(JournalError):
    """Un autre processus tient déjà ce journal en écriture. Deux processus qui reprennent le
    MÊME run refont chacun l'effet en cours : un seul écrivain à la fois."""


class JournalCorrupted(JournalError):
    """La chaîne d'empreintes du journal ne se vérifie plus (enregistrement modifié ou retiré
    au milieu du fichier). Attribut ``seq`` : le premier enregistrement fautif."""

    def __init__(self, message: str = "", *args: Any) -> None:
        super().__init__(message, *args)
        self.seq: int | None = None


class OutcomeUnknown(_ResumableError):
    """Levée à la REPRISE depuis le journal : un appel dont l'intention est écrite mais pas le
    résultat — l'effet a peut-être eu lieu (le processus est mort entre les deux).

    La relancer pourrait refaire un envoi, un paiement. Rien n'a été relancé : à toi de dire ce
    qui s'est passé, à partir de ce que SEUL le système externe sait (le mail est-il parti ?)::

        journal.resolve(call_id, ok=True, result={"envoye": True})   # il est parti
        journal.resolve(call_id, retry=True)                         # il n'est pas parti : relance
        agent.resume_from_journal(journal)

    Attributes: ``calls`` (les appels à l'issue inconnue), ``state`` (instantané), ``messages``.
    """

    def __init__(self, message: str = "", *args: Any) -> None:
        super().__init__(message, *args)
        self.calls: list[ToolCall] = []
