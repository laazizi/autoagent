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

import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .logging import get_logger
from .schema import LLMRequest, Message

if TYPE_CHECKING:  # import de type uniquement — pas de cycle à l'exécution
    from .providers.base import LLMProvider

__all__ = ["BufferMemory", "FactMemory", "Memory", "SummarizingMemory"]

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


_FACTS_SYSTEM = (
    "Tu maintiens la MÉMOIRE FACTUELLE d'un agent : une liste de faits courts, "
    "atomiques et ACTUELS (préférences, décisions, valeurs, identifiants, dates, "
    "engagements). On te donne les faits EXISTANTS (avec leur id) et de NOUVEAUX "
    'échanges. Réponds UNIQUEMENT un objet JSON {"operations": [...]} :\n'
    '- {"op": "add", "fact": "...", "subject": "..."} — fait nouveau (1 fait = '
    "1 information, court, autoporteur) ;\n"
    '- {"op": "update", "id": N, "fact": "..."} — un fait existant est contredit '
    "ou précisé par les nouveaux échanges ;\n"
    '- {"op": "delete", "id": N} — un fait n\'est plus vrai et n\'a pas de '
    "remplaçant.\n"
    "Ne crée JAMAIS un doublon d'un fait existant. Ignore le bavardage sans "
    'valeur durable. S\'il n\'y a rien à retenir : {"operations": []}.'
)


class FactMemory:
    """Mémoire FACTUELLE : des faits atomiques tenus À JOUR (0.12.0).

    Là où ``SummarizingMemory`` replie les vieux tours dans un résumé en
    prose (où une contradiction s'EMPILE), ``FactMemory`` les fait passer
    par une extraction LLM qui maintient une liste de faits courts via
    des opérations **add / update / delete** — « préfère le matin »
    REMPLACE « préfère le soir » au lieu de coexister avec. Le contexte
    injecté est la liste des faits (dense), pas des messages bruts.

    Points de design :
      * ``compact()`` borne le contexte comme ``SummarizingMemory``
        (``max_messages`` / ``keep_recent``, coupe alignée sur un message
        ``user``) ; les tours repliés passent par l'extraction. Échec de
        l'appel LLM → compaction SAUTÉE ce tour-ci (rien de tronqué en
        silence — même contrat de résilience).
      * ``remember(fait)`` ajoute un fait DIRECTEMENT (sans LLM) — c'est
        ce que branche ``agent.register_remember_tool()`` : l'agent
        mémorise volontairement, l'appel est visible dans la trace.
      * ``recall(query)`` : recherche lexicale sur les faits (courts et
        denses — le lexical y marche bien mieux que sur des messages).
      * ``path=`` : persistance JSON lisible/corrigeable à la main — un
        fichier par identité (par appelant, par client…). Effacer une
        personne = supprimer son fichier.
      * Les faits SURVIVENT aux conversations (c'est le but) : un
        historique qui raccourcit ne remet pas la base à zéro.

    Args:
        provider: LLM de l'extraction/consolidation (un modèle pas cher
            convient — même rôle que le résumeur de SummarizingMemory).
        path: fichier JSON de persistance (optionnel ; créé au premier
            fait). Sans ``path``, la base vit le temps de l'instance.
        max_messages / keep_recent: mêmes bornes que SummarizingMemory.
        max_context_facts: nombre max de faits injectés dans le contexte
            (les plus récemment mis à jour d'abord).
        max_facts: taille max de la base (au-delà, les faits les plus
            anciennement mis à jour sont écartés).
        extract_max_tokens: budget de la réponse d'extraction.
    """

    _MARKER = "[Faits mémorisés]"

    def __init__(
        self,
        provider: "LLMProvider",
        *,
        path: str | Path | None = None,
        max_messages: int = 40,
        keep_recent: int = 12,
        max_context_facts: int = 20,
        max_facts: int = 500,
        extract_max_tokens: int = 800,
    ) -> None:
        if max_messages < 2 or keep_recent < 1 or keep_recent >= max_messages:
            raise ValueError("exige max_messages >= 2 et 1 <= keep_recent < max_messages")
        if max_context_facts < 1 or max_facts < 1:
            raise ValueError("max_context_facts et max_facts doivent être >= 1")
        self.provider = provider
        self.path = Path(path) if path is not None else None
        self.max_messages = max_messages
        self.keep_recent = keep_recent
        self.max_context_facts = max_context_facts
        self.max_facts = max_facts
        self.extract_max_tokens = extract_max_tokens
        self._facts: list[dict[str, Any]] = []
        self._next_id = 1
        self._covered = 0  # nb de messages non-système déjà passés par l'extraction
        self._covered_fp = ""  # empreinte du préfixe couvert (détection de nouvelle conversation)
        if self.path is not None and self.path.exists():
            self._load()

    # ── protocole Memory ─────────────────────────────────────────────────

    def compact(self, messages: list[Message]) -> list[Message]:
        system_msgs = [m for m in messages if m.role == "system"]
        others = [m for m in messages if m.role != "system"]
        # Réabsorption : l'hôte qui persiste l'historique compacté nous
        # repasse notre message de faits in-band — on le retire (il sera
        # ré-injecté frais), la base de faits vit ailleurs (self/path).
        system_msgs = [
            m for m in system_msgs if not (m.content or "").startswith(self._MARKER)
        ]
        if self._covered > len(others) or (
            self._covered and _prefix_fingerprint(others[: self._covered]) != self._covered_fp
        ):
            # Historique raccourci OU conversation DIFFÉRENTE de longueur
            # similaire (le cas multi-appels : même mémoire, nouvel appel) :
            # le préfixe « déjà couvert » n'est pas celui qu'on a traité →
            # on repart du début du transcript. Les FAITS, eux, sont
            # conservés — c'est leur raison d'être. Trouvé par test réel :
            # sans l'empreinte, le 2e appel d'un même appelant n'était
            # jamais extrait (contradictions perdues en silence).
            self._covered = 0
            self._covered_fp = ""
        if len(others) <= self.max_messages:
            return self._assemble(system_msgs, others)
        cut = len(others) - self.keep_recent
        while cut < len(others) and others[cut].role != "user":
            cut += 1
        if cut >= len(others):  # aucun user dans la fenêtre récente — dégénéré
            cut = max(self._covered, len(others) - self.keep_recent)
        to_fold = others[self._covered : cut]
        if to_fold:
            try:
                self._extract(to_fold)
            except Exception:
                _log.exception(
                    "fact extraction failed; skipping compaction this turn (context unchanged)"
                )
                return list(messages)
            self._covered = cut
            self._covered_fp = _prefix_fingerprint(others[:cut])
        return self._assemble(system_msgs, others[cut:])

    def recall(self, query: str, k: int = 5) -> list[Message]:
        terms = {t for t in query.lower().split() if len(t) > 2}
        if not terms:
            return []
        scored = []
        for fact in self._facts:
            haystack = set(
                (fact["fact"] + " " + (fact.get("subject") or "")).lower().split()
            )
            score = len(terms & haystack)
            if score:
                scored.append((score, fact["id"]))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        by_id = {fact["id"]: fact for fact in self._facts}
        return [
            Message(role="user", content=f"[Fait #{fid}] {by_id[fid]['fact']}")
            for _, fid in scored[:k]
        ]

    # ── API factuelle ────────────────────────────────────────────────────

    def remember(self, fact: str, *, subject: str | None = None) -> dict[str, Any]:
        """Ajoute un fait DIRECTEMENT (sans appel LLM). Déduplique à
        l'identique (le fait existant est alors « touché » : sa date de
        mise à jour est rafraîchie). Retourne le fait stocké."""
        fact = (fact or "").strip()
        if not fact:
            raise ValueError("fact must be a non-empty string")
        for existing in self._facts:
            if existing["fact"].strip().lower() == fact.lower():
                existing["updated"] = self._today()
                self._save()
                return dict(existing)
        stored = self._add(fact, subject)
        self._save()
        return dict(stored)

    def forget(self, fact_id: int) -> bool:
        """Supprime un fait par id. Retourne True s'il existait."""
        before = len(self._facts)
        self._facts = [f for f in self._facts if f["id"] != fact_id]
        removed = len(self._facts) < before
        if removed:
            self._save()
        return removed

    def facts(self) -> list[dict[str, Any]]:
        """Copie de la base de faits (pour inspection/audit hôte)."""
        return [dict(f) for f in self._facts]

    # ── interne ──────────────────────────────────────────────────────────

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d")

    def _add(self, fact: str, subject: str | None) -> dict[str, Any]:
        stored = {
            "id": self._next_id,
            "fact": fact,
            "subject": (subject or "").strip() or None,
            "updated": self._today(),
        }
        self._facts.append(stored)
        self._next_id += 1
        if len(self._facts) > self.max_facts:
            # Écarte les plus anciennement mis à jour (ordre stable sinon).
            self._facts.sort(key=lambda f: f["updated"])
            self._facts = self._facts[-self.max_facts :]
            self._facts.sort(key=lambda f: f["id"])
        return stored

    def _extract(self, to_fold: list[Message]) -> None:
        lines = ["Faits existants :"]
        if self._facts:
            for fact in self._facts:
                lines.append(f"- [id {fact['id']}] {fact['fact']}")
        else:
            lines.append("(aucun)")
        lines.append("\nNouveaux échanges :")
        for message in to_fold:
            content = (message.content or "").strip()
            if len(content) > 2000:  # borne le prompt d'extraction
                content = content[:2000] + "…"
            if content:
                lines.append(f"{message.role}: {content}")
        response = self.provider.complete(
            LLMRequest(
                messages=[
                    Message(role="system", content=_FACTS_SYSTEM),
                    Message(role="user", content="\n".join(lines)),
                ],
                temperature=0,
                max_tokens=self.extract_max_tokens,
                tool_choice="none",
                response_format={"type": "json_object"},
            )
        )
        self._apply_operations(_parse_operations(response.content or ""))
        self._save()

    def _apply_operations(self, operations: list[dict[str, Any]]) -> None:
        by_id = {fact["id"]: fact for fact in self._facts}
        for op in operations:
            kind = op.get("op")
            if kind == "add":
                fact = str(op.get("fact") or "").strip()
                if fact and not any(
                    f["fact"].strip().lower() == fact.lower() for f in self._facts
                ):
                    self._add(fact, op.get("subject"))
            elif kind == "update":
                target = by_id.get(op.get("id"))
                fact = str(op.get("fact") or "").strip()
                if target is not None and fact:
                    target["fact"] = fact
                    target["updated"] = self._today()
                else:
                    _log.debug("fact update ignoré (id inconnu ou fait vide): %r", op)
            elif kind == "delete":
                if op.get("id") in by_id:
                    self._facts = [f for f in self._facts if f["id"] != op["id"]]
                    by_id.pop(op["id"], None)
            else:
                _log.debug("opération de fait inconnue ignorée: %r", op)

    def _assemble(self, system_msgs: list[Message], tail: list[Message]) -> list[Message]:
        if not self._facts:
            return [*system_msgs, *tail]
        # Les plus récemment mis à jour d'abord, bornés, ré-ordonnés par id
        # pour un rendu stable.
        chosen = sorted(self._facts, key=lambda f: f["updated"], reverse=True)
        chosen = sorted(chosen[: self.max_context_facts], key=lambda f: f["id"])
        lines = [self._MARKER]
        for fact in chosen:
            subject = f" ({fact['subject']})" if fact.get("subject") else ""
            lines.append(f"- {fact['fact']}{subject}")
        return [*system_msgs, Message(role="system", content="\n".join(lines)), *tail]

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(
                    {"facts": self._facts, "next_id": self._next_id},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            _log.exception("fact store write failed (%s); facts kept in memory", self.path)

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            facts = data.get("facts")
            if isinstance(facts, list):
                self._facts = [f for f in facts if isinstance(f, dict) and "id" in f and "fact" in f]
            self._next_id = int(data.get("next_id") or (max((f["id"] for f in self._facts), default=0) + 1))
        except (OSError, ValueError):
            _log.exception("fact store unreadable (%s); starting empty", self.path)
            self._facts, self._next_id = [], 1


def _prefix_fingerprint(messages: list[Message]) -> str:
    """Empreinte stable d'un préfixe de conversation (rôles + contenus)."""
    hasher = hashlib.sha256()
    for message in messages:
        hasher.update(message.role.encode())
        hasher.update(b"\x01")
        hasher.update((message.content or "").encode("utf-8", "replace"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def _parse_operations(content: str) -> list[dict[str, Any]]:
    """Extrait la liste d'opérations de la réponse d'extraction.

    Tolérant : fences ```json éventuelles (Anthropic est en best-effort
    JSON), objet mal formé → liste vide (la compaction du tour est alors
    un no-op factuel, jamais une erreur)."""
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
    except ValueError:
        _log.warning("fact extraction returned non-JSON; no operations applied")
        return []
    operations = data.get("operations") if isinstance(data, dict) else None
    return [op for op in operations if isinstance(op, dict)] if isinstance(operations, list) else []
