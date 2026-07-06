"""Memory abstractions for autoagent (added in 0.6.0).

A `Memory` lets a host shape what the agent sees on each call. The
protocol has two methods:

* `compact(messages)` — return a (possibly reshaped) message list for
  the upcoming provider call. Implementations may truncate the tail,
  summarize old turns, project state from artifacts, or anything
  else. Returning the input unchanged is a valid no-op. Side effects
  are allowed: a vector-backed implementation, for instance, can
  embed and index chunks here BEFORE discarding them from the working
  set, so a later `recall()` can fetch them.

* `recall(query, k)` — retrieve past messages relevant to a query.
  Used by a host-registered tool so the agent can fetch forgotten
  details on demand. Implementations that don't support semantic
  retrieval return an empty list.

The library ships one trivial implementation, ``BufferMemory`` (keep
the last N non-system messages, drop the rest). Richer
implementations — vector-backed semantic memory, recursive
summarisation, code-state projection — live in ``examples/`` and pull
in their own opinionated stack (embedding provider, vector store).
The lib stays under its auditability budget and the host picks the
backend that fits.

Agent integration is intentionally minimal: ``Agent.run_messages``
calls ``memory.compact(messages)`` ONCE at the start of the run,
before the loop. This keeps the existing ``turn_start`` / post-turn
hook accounting simple and predictable. Hosts that need finer-grained
consolidation (mid-run compaction) can call ``memory.compact`` from
their own code between turns and pass the result back in.

Threading: implementations should be safe under one caller at a time
(same contract as ``Agent``). Concurrent runs across threads should
use separate ``Memory`` instances.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .logging import get_logger
from .schema import LLMRequest, Message

if TYPE_CHECKING:  # import de type uniquement — pas de cycle à l'exécution
    from .providers.base import LLMProvider

__all__ = ["BufferMemory", "Memory", "SummarizingMemory"]

_log = get_logger("memory")


@runtime_checkable
class Memory(Protocol):
    """Protocol implemented by anything that wants to shape the agent's view.

    Two methods, both invoked by the host (the agent itself only calls
    ``compact``). Implementations are free to do side effects — log,
    embed, index, persist — as long as ``compact`` returns a coherent
    message list and ``recall`` returns past ``Message`` objects.
    """

    def compact(self, messages: list[Message]) -> list[Message]:
        """Return a reshaped message list for the next provider call.

        Implementations decide whether to truncate, summarise, project,
        or do nothing. The returned list MUST keep the conversation
        well-formed for downstream providers — in particular, any
        ``tool`` message must still follow an ``assistant`` message
        with the matching ``tool_call_id``.
        """
        ...

    def recall(self, query: str, k: int = 5) -> list[Message]:
        """Retrieve past messages relevant to ``query``.

        Used by the host-registered ``recall`` tool so the agent can
        explicitly fetch forgotten details. Implementations without
        semantic retrieval may safely return an empty list.
        """
        ...


class BufferMemory:
    """Trivial Memory: keep system messages + at most ``max_messages`` non-system.

    No dependencies, no LLM calls, no embeddings. Useful as a baseline
    and as a starting point for chat apps that don't need semantic
    retrieval.

    Hard cap: the returned message list contains AT MOST ``max_messages``
    non-system messages, always. The cap is honoured even when the input
    is malformed (orphan ``tool`` messages, missing user message, ...).

    Well-formedness: the returned tail starts at the first ``user``
    message inside the budget, so we never leave an orphan ``tool``
    message at the front (which strict providers reject). If the
    ``max_messages``-sized tail contains no user message, the tail is
    dropped entirely — better to send a tight system-only context than
    a malformed conversation.
    """

    def __init__(self, max_messages: int = 20) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be >= 1")
        self.max_messages = max_messages

    def compact(self, messages: list[Message]) -> list[Message]:
        system_msgs = [m for m in messages if m.role == "system"]
        others = [m for m in messages if m.role != "system"]
        if not others:
            return list(messages)
        # Hard cap: take at most max_messages from the tail. This bound
        # is non-negotiable — the cost of including one more message is
        # the caller's to make, not ours.
        tail = others[-self.max_messages :]
        # Well-formedness: walk forward to the first ``user`` message
        # so we never lead with an orphan ``tool`` / ``assistant``
        # message. If no user message exists in our budget, drop the
        # tail rather than break the cap to fetch one.
        first_user = next((i for i, m in enumerate(tail) if m.role == "user"), None)
        tail = tail[first_user:] if first_user is not None else []
        return [*system_msgs, *tail]

    def recall(self, query: str, k: int = 5) -> list[Message]:
        return []


_SUMMARY_SYSTEM = (
    "Tu compresses l'historique d'une conversation agent<->outils. Produis un "
    "résumé DENSE et FACTUEL qui préserve : les décisions prises, les faits et "
    "valeurs établis (chiffres, chemins, identifiants, URLs), les résultats "
    "d'outils importants, les préférences exprimées par l'utilisateur, et ce "
    "qui reste à faire. Pas de préambule, pas de conclusion — uniquement le "
    "résumé. Fusionne le résumé précédent (s'il y en a un) avec les nouveaux "
    "échanges en UN seul résumé cohérent."
)


class SummarizingMemory:
    """Memory qui RÉSUME les tours anciens au lieu de les jeter (0.10.0).

    Là où ``BufferMemory`` tronque (les vieux tours disparaissent),
    ``SummarizingMemory`` replie les tours au-delà de ``max_messages``
    dans un résumé LLM injecté comme message ``system`` — le contexte
    reste borné SANS perdre les décisions/valeurs établies. Le résumé
    est INCRÉMENTAL : chaque compaction ne résume que les tours pas
    encore couverts (fusionnés avec le résumé précédent), donc UN appel
    LLM par compaction, pas une re-synthèse de tout l'historique.

    Sécurité d'échec : si l'appel de résumé échoue (réseau, quota), la
    compaction est SAUTÉE ce tour-ci — les messages repartent inchangés
    (le contexte grossit temporairement) plutôt que d'être tronqués en
    silence. L'erreur est loguée, jamais propagée (même contrat de
    résilience que les autres callables hôtes).

    ``recall(query)`` fait une recherche LEXICALE (recouvrement de
    termes, zéro dépendance — pas d'embeddings) dans les messages déjà
    repliés : brancher ``agent.register_recall_tool()`` permet à
    l'agent de retrouver un détail sorti de sa fenêtre.

    Args:
        provider: le LLM qui rédige les résumés (peut être un modèle
            moins cher que celui de l'agent).
        max_messages: au-delà de ce nombre de messages non-système, on
            compacte.
        keep_recent: nombre de messages récents gardés VERBATIM (la
            coupe est alignée sur un message ``user`` pour ne jamais
            laisser un ``tool`` orphelin en tête).
        summary_max_tokens: budget du résumé.
    """

    def __init__(
        self,
        provider: "LLMProvider",
        *,
        max_messages: int = 40,
        keep_recent: int = 12,
        summary_max_tokens: int = 600,
    ) -> None:
        if max_messages < 2 or keep_recent < 1 or keep_recent >= max_messages:
            raise ValueError("exige max_messages >= 2 et 1 <= keep_recent < max_messages")
        self.provider = provider
        self.max_messages = max_messages
        self.keep_recent = keep_recent
        self.summary_max_tokens = summary_max_tokens
        self._summary = ""
        self._covered = 0  # nb de messages non-système déjà repliés dans le résumé
        self._archive: list[Message] = []  # tout ce qui a été replié (pour recall)

    _MARKER = "[Résumé de la conversation antérieure]"

    def compact(self, messages: list[Message]) -> list[Message]:
        system_msgs = [m for m in messages if m.role == "system"]
        others = [m for m in messages if m.role != "system"]
        # Un hôte qui persiste l'historique COMPACTÉ (le pattern courant :
        # sauvegarder result.messages) nous repasse notre propre résumé comme
        # message système in-band. On le réabsorbe comme graine au lieu d'en
        # empiler un deuxième.
        inband = [m for m in system_msgs if (m.content or "").startswith(self._MARKER)]
        if inband:
            system_msgs = [m for m in system_msgs if m not in inband]
            if not self._summary:
                self._summary = inband[-1].content[len(self._MARKER) :].strip()
        if self._covered > len(others):
            # L'historique a raccourci : soit l'hôte nous repasse un historique
            # DÉJÀ compacté (résumé in-band réabsorbé ci-dessus -> on le garde),
            # soit c'est une nouvelle conversation (pas de marqueur -> zéro).
            self._covered, self._archive = 0, []
            if not inband:
                self._summary = ""
        if len(others) <= self.max_messages:
            return self._assemble(system_msgs, others[self._covered :])
        # Coupe : garder keep_recent messages, alignée sur un ``user``.
        cut = len(others) - self.keep_recent
        while cut < len(others) and others[cut].role != "user":
            cut += 1
        if cut >= len(others):  # aucun user dans la fenêtre récente — dégénéré
            cut = max(self._covered, len(others) - self.keep_recent)
        to_fold = others[self._covered : cut]
        if to_fold:
            try:
                self._summary = self._summarize(to_fold)
            except Exception:
                _log.exception(
                    "summarize failed; skipping compaction this turn (context unchanged)"
                )
                return list(messages)
            self._archive.extend(to_fold)
            self._covered = cut
        return self._assemble(system_msgs, others[cut:])

    def recall(self, query: str, k: int = 5) -> list[Message]:
        terms = {t for t in query.lower().split() if len(t) > 2}
        if not terms:
            return []
        scored = []
        for index, message in enumerate(self._archive):
            words = set((message.content or "").lower().split())
            score = len(terms & words)
            if score:
                scored.append((score, index))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        picked = sorted(index for _, index in scored[:k])
        return [self._archive[i] for i in picked]

    def _assemble(self, system_msgs: list[Message], tail: list[Message]) -> list[Message]:
        if not self._summary:
            return [*system_msgs, *tail]
        summary_msg = Message(
            role="system",
            content="[Résumé de la conversation antérieure]\n" + self._summary,
        )
        return [*system_msgs, summary_msg, *tail]

    def _summarize(self, to_fold: list[Message]) -> str:
        lines = []
        if self._summary:
            lines.append(f"Résumé précédent :\n{self._summary}\n\nNouveaux échanges :")
        for message in to_fold:
            content = (message.content or "").strip()
            if len(content) > 2000:  # borne le prompt de résumé
                content = content[:2000] + "…"
            if content:
                lines.append(f"{message.role}: {content}")
        response = self.provider.complete(
            LLMRequest(
                messages=[
                    Message(role="system", content=_SUMMARY_SYSTEM),
                    Message(role="user", content="\n".join(lines)),
                ],
                temperature=0,
                max_tokens=self.summary_max_tokens,
                tool_choice="none",
            )
        )
        return (response.content or "").strip() or self._summary
