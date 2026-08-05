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
import math
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from .logging import get_logger
from .schema import TAINT_SENTINEL, LLMRequest, Message, TokenUsage, is_tainted

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
        self._tainted = False  # 0.17 : préserve la teinte à travers la compaction
        self.last_usage: TokenUsage | None = None  # 0.17 : coût du dernier compact

    _MARKER = "[Résumé de la conversation antérieure]"

    def compact(self, messages: list[Message]) -> list[Message]:
        self.last_usage = None  # coût mesuré de CE compact (lu par la boucle pour le budget)
        system_msgs = [m for m in messages if m.role == "system"]
        others = [m for m in messages if m.role != "system"]
        # Un hôte qui persiste l'historique COMPACTÉ (le pattern courant :
        # sauvegarder result.messages) nous repasse notre propre résumé comme
        # message système in-band. On le réabsorbe comme graine au lieu d'en
        # empiler un deuxième.
        inband = [m for m in system_msgs if (m.content or "").startswith(self._MARKER)]
        if inband:
            system_msgs = [m for m in system_msgs if m not in inband]
            if TAINT_SENTINEL in (inband[-1].content or ""):
                self._tainted = True  # réabsorbe la teinte d'un historique persisté
            if not self._summary:
                self._summary = inband[-1].content[len(self._MARKER) :].split(TAINT_SENTINEL)[0].strip()
        # Teinte : dès qu'on replie du contenu externe non fiable, on la retient
        # (elle survivra dans le message de résumé via la sentinelle).
        if is_tainted(others):
            self._tainted = True
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
        suffixe = f"\n{TAINT_SENTINEL}" if self._tainted else ""
        if not self._summary:
            if self._tainted:  # teinte sans résumé encore : on la porte quand même
                return [*system_msgs, Message(role="system", content=self._MARKER + suffixe), *tail]
            return [*system_msgs, *tail]
        summary_msg = Message(
            role="system",
            content=self._MARKER + "\n" + self._summary + suffixe,
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
        # `getattr` : la comptabilité de jetons est un BONUS, pas une exigence. Un
        # provider tiers ou un double de test qui renvoie un objet sans `usage` ne
        # doit pas faire échouer une compaction — elle est best-effort par contrat.
        self.last_usage = getattr(response, "usage", None)
        return (response.content or "").strip() or self._summary


_FORGET_SYSTEM = (
    "Tu appliques une demande d'OUBLI sur la mémoire factuelle d'un agent. On te "
    "donne une INSTRUCTION en langue naturelle et une liste de FAITS numérotés.\n"
    'Réponds UNIQUEMENT par un objet JSON : {"forget": [ids], "reason": "..."}.\n'
    "Règles STRICTES :\n"
    "- ne mets dans `forget` que les ids des faits RÉELLEMENT visés par "
    "l'instruction ;\n"
    "- attention aux COLLISIONS de préfixe : « Paul Martin » n'est pas "
    "« Paul Martineau » ; « dossier 12 » n'est pas « dossier 120 » ;\n"
    "- reconnais les VARIANTES d'un même identifiant (espaces, tirets, casse, "
    "formulation dans une autre langue) ;\n"
    "- un fait COMPOSÉ qui ne concerne l'instruction que PARTIELLEMENT ne doit "
    "PAS être supprimé (le signaler dans `reason`) ;\n"
    "- en cas de DOUTE sur un fait, ne le supprime pas. Mieux vaut oublier trop "
    "peu que détruire une donnée juste.\n"
    'Si rien ne correspond, renvoie {"forget": [], "reason": "aucune correspondance"}.'
)


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
        background: consolidation en ARRIÈRE-PLAN (« sleep-time », 0.13) —
            l'appel LLM d'extraction sort du chemin critique : ``compact()``
            lance l'extraction dans un thread et rend la main immédiatement
            (les vieux tours restent dans le contexte UN tour de plus, puis
            sont repliés une fois les faits sauvés — jamais de troncature
            avant sauvegarde). ``flush()`` attend la fin (tests, arrêt).
        embed_fn: fonction d'embedding OPTIONNELLE ``list[str] ->
            list[list[float]]`` fournie par l'hôte (API OpenAI/Gemini,
            modèle local…). Quand elle est fournie, ``recall`` cherche par
            SENS (cosinus) au lieu du lexical — « véhicule » retrouve
            « voiture ». Embeddings calculés paresseusement au premier
            recall (un lot), persistés dans un fichier annexe
            ``<path>.vectors.json`` (le JSON des faits reste lisible).
            Échec d'embedding → repli lexical, jamais d'erreur.
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
        background: bool = False,
        embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
        max_consolidation_facts: int = 30,
        recall_mode: str = "hybrid",
    ) -> None:
        if max_messages < 2 or keep_recent < 1 or keep_recent >= max_messages:
            raise ValueError("exige max_messages >= 2 et 1 <= keep_recent < max_messages")
        if max_context_facts < 1 or max_facts < 1 or max_consolidation_facts < 1:
            raise ValueError(
                "max_context_facts, max_facts et max_consolidation_facts doivent être >= 1"
            )
        self.provider = provider
        self.path = Path(path) if path is not None else None
        self.max_messages = max_messages
        self.keep_recent = keep_recent
        self.max_context_facts = max_context_facts
        self.max_facts = max_facts
        self.extract_max_tokens = extract_max_tokens
        self.background = background
        self.embed_fn = embed_fn
        self.max_consolidation_facts = max_consolidation_facts
        if recall_mode not in ("hybrid", "lexical", "semantic"):
            raise ValueError("recall_mode doit valoir 'hybrid', 'lexical' ou 'semantic'")
        self.recall_mode = recall_mode
        self._facts: list[dict[str, Any]] = []
        self._next_id = 1
        self._covered = 0  # nb de messages non-système déjà passés par l'extraction
        self._covered_fp = ""  # empreinte du préfixe couvert (détection de nouvelle conversation)
        self._lock = threading.Lock()  # garde _facts/_next_id (worker + hôte)
        self._job: dict[str, Any] | None = None  # extraction en cours (mode background)
        self._vectors: dict[int, list[float]] = {}  # id de fait -> embedding
        self._tainted = False  # 0.17 : préserve la teinte à travers la compaction
        self.last_usage: TokenUsage | None = None  # 0.17 : coût de la dernière extraction
        if self.path is not None and self.path.exists():
            self._load()

    # ── protocole Memory ─────────────────────────────────────────────────

    def compact(self, messages: list[Message]) -> list[Message]:
        self.last_usage = None  # coût de CE compact (sync) ; en background, renseigné plus tard
        system_msgs = [m for m in messages if m.role == "system"]
        others = [m for m in messages if m.role != "system"]
        # Réabsorption : l'hôte qui persiste l'historique compacté nous
        # repasse notre message de faits in-band — on le retire (il sera
        # ré-injecté frais), la base de faits vit ailleurs (self/path).
        if any((m.content or "").startswith(self._MARKER) and TAINT_SENTINEL in (m.content or "")
               for m in system_msgs):
            self._tainted = True  # réabsorbe la teinte d'un historique persisté
        system_msgs = [
            m for m in system_msgs if not (m.content or "").startswith(self._MARKER)
        ]
        # Teinte : dès qu'on voit du contenu externe non fiable, on la retient
        # (portée ensuite par le message [Faits mémorisés] via la sentinelle).
        if is_tainted(others):
            self._tainted = True
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
        # Mode background : une extraction lancée à un tour précédent vient
        # de finir ? On n'adopte le repli qu'ICI, une fois les faits SAUVÉS —
        # jamais de troncature avant sauvegarde.
        if self._job is not None and not self._job["thread"].is_alive():
            job, self._job = self._job, None
            if (
                job.get("error") is None
                and job["cut"] <= len(others)
                and _prefix_fingerprint(others[: job["cut"]]) == job["fp"]
            ):
                self._covered = job["cut"]
                self._covered_fp = job["fp"]
            # échec ou transcript changé → la tranche sera retentée telle
            # qu'elle est aujourd'hui ; rien n'a été perdu.
        if len(others) <= self.max_messages:
            return self._assemble(system_msgs, others)
        cut = len(others) - self.keep_recent
        while cut < len(others) and others[cut].role != "user":
            cut += 1
        if cut >= len(others):  # aucun user dans la fenêtre récente — dégénéré
            cut = max(self._covered, len(others) - self.keep_recent)
        to_fold = list(others[self._covered : cut])
        if to_fold and self.background:
            if self._job is None:
                # « Sleep-time » (0.13) : l'appel LLM part dans un thread,
                # la conversation ne l'attend JAMAIS. Le contexte reste
                # entier un tour de plus (coût borné et connu) ; le repli
                # aura lieu au prochain compact(), après sauvegarde.
                job: dict[str, Any] = {
                    "cut": cut,
                    "fp": _prefix_fingerprint(others[:cut]),
                    "error": None,
                }

                def _worker() -> None:
                    try:
                        self._extract(to_fold)
                    except Exception as exc:  # noqa: BLE001
                        job["error"] = exc
                        _log.exception(
                            "background fact extraction failed; slice will be retried"
                        )

                job["thread"] = threading.Thread(
                    target=_worker, name="autoagent-factmemory", daemon=True
                )
                self._job = job
                job["thread"].start()
            return self._assemble(system_msgs, others[self._covered :])
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

    def flush(self, timeout: float | None = None) -> bool:
        """Attend la fin d'une consolidation en arrière-plan (arrêt propre,
        tests). Retourne ``False`` si le délai expire. No-op en mode
        synchrone."""
        job = self._job
        if job is None:
            return True
        job["thread"].join(timeout)
        return not job["thread"].is_alive()

    def recall(self, query: str, k: int = 5) -> list[Message]:
        """Retrouve les faits pertinents. Voir `recall_mode` (0.18.0).

        `hybrid` (défaut) fusionne un classement LEXICAL (BM25) et un classement
        SÉMANTIQUE (cosinus, si `embed_fn`) par RRF. Les deux signaux échouent sur
        des requêtes opposées : le sémantique perd les correspondances exactes
        (n° de contrat, SIREN, plaque, identifiant), le lexical perd les
        synonymes. Les fusionner rattrape les deux angles morts ; sans `embed_fn`,
        BM25 tourne seul et reste très supérieur à l'ancienne intersection de mots
        (qui n'avait ni IDF, ni normalisation de longueur, ni tokenisation).
        """
        with self._lock:
            # Faits COURANTS uniquement : on ne sert jamais un fait périmé comme
            # s'il était vrai (le mode d'échec n°1 des mémoires d'agent).
            snapshot = [dict(f) for f in self._valid()]
        if not snapshot:
            return []

        lexical = self._bm25_rank(query, snapshot)
        semantic: list[dict[str, Any]] = []
        if self.recall_mode in ("hybrid", "semantic") and self.embed_fn is not None:
            try:
                semantic = self._semantic_rank(query, snapshot)
            except Exception:
                # Contrat inchangé : un embed_fn en panne ne casse pas le recall.
                _log.exception("embed_fn failed; falling back to lexical recall")
                semantic = []

        if self.recall_mode == "lexical" or not semantic:
            ranked = lexical
        elif self.recall_mode == "semantic":
            ranked = semantic
        else:
            ranked = _rrf_fuse(lexical, semantic)

        return [
            Message(role="user", content=f"[Fait #{fact['id']}] {fact['fact']}")
            for fact in ranked[:k]
        ]

    def _bm25_rank(self, query: str, snapshot: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Classement BM25 (Okapi) sur `fact` + `subject`.

        De l'arithmétique pure — aucune dépendance, aucun appel réseau. Deux
        propriétés que l'intersection de mots n'avait pas : l'IDF (un terme rare
        pèse plus qu'un mot passe-partout) et la saturation/normalisation de
        longueur (un fait court et ciblé n'est pas noyé par un fait bavard).
        """
        terms = _tokenize(query)
        if not terms:
            return []
        docs = [
            (fact, _tokenize(f"{fact['fact']} {fact.get('subject') or ''}"))
            for fact in snapshot
        ]
        n = len(docs)
        avg_len = sum(len(tokens) for _, tokens in docs) / n if n else 0.0
        if not avg_len:
            return []
        k1, b = 1.5, 0.75
        doc_freq: dict[str, int] = {}
        for _, tokens in docs:
            for term in set(tokens):
                doc_freq[term] = doc_freq.get(term, 0) + 1
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for fact, tokens in docs:
            if not tokens:
                continue
            score = 0.0
            for term in terms:
                tf = tokens.count(term)
                if not tf:
                    continue
                df = doc_freq.get(term, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                norm = tf + k1 * (1 - b + b * len(tokens) / avg_len)
                score += idf * (tf * (k1 + 1)) / norm
            if score > 0:
                scored.append((-score, fact["id"], fact))
        scored.sort(key=lambda item: (item[0], item[1]))  # score desc, id asc (stable)
        return [fact for _, _, fact in scored]

    def _semantic_rank(self, query: str, snapshot: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Classement par cosinus d'embeddings (liste vide = pas de signal)."""
        missing = [f for f in snapshot if f["id"] not in self._vectors]
        if missing:
            vectors = self.embed_fn([f["fact"] for f in missing])
            with self._lock:
                for fact, vector in zip(missing, vectors):
                    self._vectors[fact["id"]] = list(vector)
            self._save_vectors()
        query_vec = self.embed_fn([query])[0]
        with self._lock:
            pairs = [(fact, self._vectors.get(fact["id"])) for fact in snapshot]
        scored = [
            (-_cosine(query_vec, vector), fact["id"], fact)
            for fact, vector in pairs
            if vector is not None
        ]
        scored = [item for item in scored if item[0] < 0]  # cosinus > 0
        scored.sort(key=lambda item: (item[0], item[1]))
        return [fact for _, _, fact in scored]

    # ── API factuelle ────────────────────────────────────────────────────

    def remember(self, fact: str, *, subject: str | None = None,
                 source: str = "host") -> dict[str, Any]:
        """Ajoute un fait DIRECTEMENT (sans appel LLM). Déduplique à
        l'identique (le fait existant est alors « touché » : sa date de
        mise à jour est rafraîchie). Retourne le fait stocké."""
        fact = (fact or "").strip()
        if not fact:
            raise ValueError("fact must be a non-empty string")
        with self._lock:
            for existing in self._valid():
                if existing["fact"].strip().lower() == fact.lower():
                    existing["updated"] = self._today()
                    self._save()
                    return dict(existing)
            stored = self._add(fact, subject, source=source)
            self._save()
            return dict(stored)

    def forget(self, fact_id: int) -> bool:
        """Supprime un fait par id. Retourne True s'il existait."""
        with self._lock:
            before = len(self._facts)
            self._facts = [f for f in self._facts if f["id"] != fact_id]
            self._vectors.pop(fact_id, None)
            removed = len(self._facts) < before
            if removed:
                self._save()
            return removed

    def forget_matching(self, instruction: str, *, dry_run: bool = False) -> list[dict[str, Any]]:
        """Oublie les faits désignés en LANGUE NATURELLE (0.18.0).

        « oublie tout ce qui concerne mon ancien employeur », « supprime les
        données liées au dossier X ». Renvoie la liste des faits supprimés (copie
        complète, pour la trace et la preuve d'effacement).

        Pourquoi ça ne se réduit pas à ``forget(id)`` : jusqu'ici la seule
        décision confiée au LLM était l'ÉCRITURE (extraction/consolidation dans
        ``compact``). Or les architectures « décision à l'écriture seule » —
        celle-ci — échouent sur la suppression INTENTIONNELLE : collision de
        préfixe (« Paul Martin » vs « Paul Martineau »), faits composés (« il
        travaille chez X et aime le thé » : n'oublier que l'employeur), variantes
        d'identifiants, formulations d'une autre langue. Déplacer la décision au
        moment de la MUTATION récupère ces cas — c'est le gain unitaire le plus
        élevé de l'état de l'art 2026 sur l'oubli.

        Le chemin de LECTURE n'est pas ralenti : ce coût (un appel LLM) n'est payé
        qu'ici, quand un humain ou l'agent demande explicitement un oubli.

        `dry_run=True` renvoie ce qui SERAIT supprimé sans rien toucher — à
        utiliser pour faire confirmer un effacement avant de l'appliquer.
        """
        instruction = (instruction or "").strip()
        if not instruction:
            raise ValueError("instruction must be a non-empty string")
        with self._lock:
            snapshot = [dict(f) for f in self._valid()]
        if not snapshot:
            return []

        # Pré-filtre hybride : sur une grosse base, ne soumettre au LLM que les
        # faits plausibles (même logique que `_relevant_for` pour la consolidation).
        candidats = snapshot
        if len(snapshot) > self.max_consolidation_facts:
            pertinents = self._bm25_rank(instruction, snapshot)[: self.max_consolidation_facts]
            candidats = pertinents or snapshot[: self.max_consolidation_facts]

        listing = "\n".join(f"{f['id']}: {f['fact']}" for f in candidats)
        try:
            response = self.provider.complete(
                LLMRequest(
                    messages=[
                        Message(role="system", content=_FORGET_SYSTEM),
                        Message(role="user",
                                content=f"INSTRUCTION D'OUBLI :\n{instruction}\n\n"
                                        f"FAITS :\n{listing}"),
                    ],
                    temperature=0,
                    max_tokens=self.extract_max_tokens,
                    tool_choice="none",
                    response_format={"type": "json_object"},
                )
            )
        except Exception:
            # Fail-CLOSED sur une SUPPRESSION : en cas de doute on ne supprime
            # rien (contrairement à la compaction, best-effort par contrat).
            _log.exception("forget_matching: appel LLM échoué — aucune suppression")
            return []
        self.last_usage = getattr(response, "usage", None)

        ids = _parse_forget_ids(response.content or "")
        vises = {f["id"] for f in candidats}          # jamais hors du lot soumis
        a_supprimer = [f for f in snapshot if f["id"] in ids and f["id"] in vises]
        if dry_run or not a_supprimer:
            return [dict(f) for f in a_supprimer]
        with self._lock:
            cibles = {f["id"] for f in a_supprimer}
            self._facts = [f for f in self._facts if f["id"] not in cibles]
            for fid in cibles:
                self._vectors.pop(fid, None)
            self._save()
        if self._vectors_path() is not None:
            self._save_vectors()
        return [dict(f) for f in a_supprimer]

    def facts(self, *, include_invalid: bool = False) -> list[dict[str, Any]]:
        """Copie de la base de faits (pour inspection/audit hôte).

        Par défaut : uniquement les faits COURANTS — ce que voyaient déjà les
        consommateurs avant la bi-temporalité (0.18.0), puisque `update`
        écrasait alors le texte en place. `include_invalid=True` ajoute les faits
        périmés (fenêtre fermée), pour un audit ou un « qu'est-ce qu'on croyait,
        et quand ? ».
        """
        with self._lock:
            source = self._facts if include_invalid else self._valid()
            return [dict(f) for f in source]

    def history(self, fact_id: int) -> list[dict[str, Any]]:
        """Chaîne de supersession d'un fait, du plus ancien au plus récent (0.18.0).

        Répond à « depuis quand ? » et « qu'est-ce qui a remplacé quoi ? ». On
        remonte d'abord jusqu'à la racine (le fait que personne n'a supersédé),
        puis on redescend la chaîne. Tolérant aux trous : `forget()` supprime
        DUREMENT (droit à l'effacement), ce qui peut laisser un pointeur
        pendant — la chaîne s'arrête alors proprement.
        """
        with self._lock:
            par_id = {f["id"]: dict(f) for f in self._facts}
            remplace_par = {  # qui supersède qui → pour remonter à la racine
                f["superseded_by"]: f["id"]
                for f in self._facts
                if f.get("superseded_by") is not None
            }
        if fact_id not in par_id:
            return []
        racine = fact_id
        vus = {racine}
        while racine in remplace_par and remplace_par[racine] not in vus:
            racine = remplace_par[racine]
            vus.add(racine)
        chaine, courant = [], racine
        vus = {courant}
        while courant is not None and courant in par_id:
            chaine.append(par_id[courant])
            suivant = par_id[courant].get("superseded_by")
            if suivant in vus:  # garde anti-cycle (fichier trafiqué)
                break
            vus.add(suivant)
            courant = suivant
        return chaine

    # ── interne ──────────────────────────────────────────────────────────

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d")

    def _add(self, fact: str, subject: str | None, source: str = "agent") -> dict[str, Any]:
        today = self._today()
        stored = {
            "id": self._next_id,
            "fact": fact,
            "subject": (subject or "").strip() or None,
            "updated": today,
            # ── bi-temporalité + provenance (0.18.0) ────────────────────────
            # `source` : qui l'affirme. Une déclaration de l'utilisateur ne vaut
            # pas une inférence de l'agent ; sans ce champ, impossible d'arbitrer.
            "source": source if source in _SOURCES else "agent",
            # `valid_from` / `invalid_at` : depuis quand c'est vrai DANS LE MONDE.
            # `invalid_at=None` = valide aujourd'hui. Une contradiction FERME la
            # fenêtre au lieu de détruire le fait — on ne sert jamais un fait
            # périmé comme courant, mais on peut encore dire ce qu'on croyait.
            "valid_from": today,
            "invalid_at": None,
            "superseded_by": None,
        }
        self._facts.append(stored)
        self._next_id += 1
        self._evict()
        return stored

    def _evict(self) -> None:
        """Borne la base à `max_facts`. Appelé sous self._lock.

        Ordre d'éviction : les faits PÉRIMÉS d'abord (ils ne peuvent plus être
        remontés par `recall`, ils ne servent que l'historique), puis les plus
        anciennement mis à jour. Sans cette priorité, la bi-temporalité aurait
        évincé des faits COURANTS pour garder des périmés.
        """
        if len(self._facts) <= self.max_facts:
            return
        # clé de tri : (encore valide ?, date de mise à jour) — on coupe la TÊTE
        self._facts.sort(key=lambda f: (f.get("invalid_at") is None, f.get("updated") or ""))
        retires = self._facts[: len(self._facts) - self.max_facts]
        self._facts = self._facts[len(self._facts) - self.max_facts :]
        for retire in retires:
            self._vectors.pop(retire["id"], None)
        self._facts.sort(key=lambda f: f["id"])

    def _extract(self, to_fold: list[Message]) -> None:
        # Peut tourner dans le worker : instantané des faits sous verrou,
        # appel LLM HORS verrou, application des opérations sous verrou.
        with self._lock:
            existants = [(f["id"], f["fact"]) for f in self._valid()]
        existants = self._relevant_for(to_fold, existants)
        lines = ["Faits existants :"]
        if existants:
            for fid, texte in existants:
                lines.append(f"- [id {fid}] {texte}")
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
        # `getattr` : la comptabilité de jetons est un BONUS, pas une exigence. Un
        # provider tiers ou un double de test qui renvoie un objet sans `usage` ne
        # doit pas faire échouer une compaction — elle est best-effort par contrat.
        self.last_usage = getattr(response, "usage", None)
        with self._lock:
            self._apply_operations(_parse_operations(response.content or ""))
            self._save()

    def _relevant_for(
        self, to_fold: list[Message], existants: list[tuple[int, str]]
    ) -> list[tuple[int, str]]:
        """Présélectionne les faits PERTINENTS pour la tranche à consolider.

        Envoyer TOUTE la base au LLM coûte linéairement (500 faits ≈ 15k
        tokens PAR consolidation) et dégrade sa précision — les systèmes
        de référence (Mem0) ne consolident que contre les faits similaires.
        Filtre par recouvrement lexical avec la tranche, VOLONTAIREMENT
        généreux (``max_consolidation_facts``, défaut 30) : un fait non
        montré ne peut pas être contredit — et la déduplication de
        ``_apply_operations`` (qui compare à TOUTE la base) reste le filet
        contre les doublons.
        """
        if len(existants) <= self.max_consolidation_facts:
            return existants  # petite base : comportement inchangé

        def stems(texte: str) -> set[str]:
            # Racines grossières (5 premiers caractères) : « rappel » et
            # « rappelé » doivent matcher — le filtre est un rappel large,
            # pas une recherche exacte.
            return {t[:5] for t in texte.lower().split() if len(t) > 3}

        slice_stems: set[str] = set()
        for message in to_fold:
            slice_stems |= stems(message.content or "")
        scored = []
        for index, (fid, texte) in enumerate(existants):
            overlap = len(slice_stems & stems(texte))
            scored.append((-overlap, index, fid, texte))
        scored.sort()
        kept = scored[: self.max_consolidation_facts]
        kept.sort(key=lambda item: item[1])  # ordre d'origine (stabilité du prompt)
        return [(fid, texte) for _, _, fid, texte in kept]

    def _apply_operations(self, operations: list[dict[str, Any]]) -> None:
        # Appelé sous self._lock.
        by_id = {fact["id"]: fact for fact in self._facts}
        for op in operations:
            kind = op.get("op")
            if kind == "add":
                fact = str(op.get("fact") or "").strip()
                if fact and not any(
                    f["fact"].strip().lower() == fact.lower() for f in self._valid()
                ):
                    self._add(fact, op.get("subject"))
            elif kind == "update":
                target = by_id.get(op.get("id"))
                fact = str(op.get("fact") or "").strip()
                if target is None or not fact:
                    _log.debug("fact update ignoré (id inconnu ou fait vide): %r", op)
                elif target.get("invalid_at") is not None:
                    # Opération périmée : le fait a déjà été supersédé dans ce
                    # même lot. On ne chaîne pas sur un fait mort.
                    _log.debug("fact update ignoré (fait déjà périmé): %r", op)
                else:
                    self._supersede(target, fact, op.get("subject"))
            elif kind == "delete":
                if op.get("id") in by_id:
                    self._facts = [f for f in self._facts if f["id"] != op["id"]]
                    self._vectors.pop(op["id"], None)
                    by_id.pop(op["id"], None)
            else:
                _log.debug("opération de fait inconnue ignorée: %r", op)

    def _supersede(self, ancien: dict[str, Any], nouveau_texte: str,
                   subject: str | None = None) -> dict[str, Any]:
        """Remplace un fait en FERMANT sa fenêtre de validité (0.18.0).

        Appelé sous self._lock. L'ancien comportement écrasait le texte en place :
        une extraction LLM ratée détruisait donc silencieusement une donnée juste,
        et il devenait impossible de répondre « depuis quand ? ». Ici l'ancien fait
        reste, marqué périmé et pointant vers son successeur.

        Son embedding est purgé : un fait périmé ne peut plus être remonté par
        `recall`, garder son vecteur ne servirait qu'à occuper de la place.
        """
        remplacant = self._add(nouveau_texte, subject if subject is not None
                               else ancien.get("subject"), source="agent")
        ancien["invalid_at"] = self._today()
        ancien["superseded_by"] = remplacant["id"]
        ancien["updated"] = self._today()
        self._vectors.pop(ancien["id"], None)
        return remplacant

    def _valid(self) -> list[dict[str, Any]]:
        """Faits COURANTS (fenêtre de validité ouverte). Appelé sous self._lock."""
        return [f for f in self._facts if f.get("invalid_at") is None]

    def _assemble(self, system_msgs: list[Message], tail: list[Message]) -> list[Message]:
        with self._lock:
            facts_now = [dict(f) for f in self._valid()]
        if not facts_now and not self._tainted:
            return [*system_msgs, *tail]
        # Les plus récemment mis à jour d'abord, bornés, ré-ordonnés par id
        # pour un rendu stable.
        chosen = sorted(facts_now, key=lambda f: f["updated"], reverse=True)
        chosen = sorted(chosen[: self.max_context_facts], key=lambda f: f["id"])
        lines = [self._MARKER]
        for fact in chosen:
            subject = f" ({fact['subject']})" if fact.get("subject") else ""
            lines.append(f"- {fact['fact']}{subject}")
        if self._tainted:  # porte la teinte pour qu'elle survive à la persistance
            lines.append(TAINT_SENTINEL)
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

    def _vectors_path(self) -> Path | None:
        # Fichier ANNEXE : le JSON des faits reste lisible par un humain,
        # les embeddings (gros et opaques) vivent à côté.
        return None if self.path is None else self.path.with_name(self.path.name + ".vectors.json")

    def _save_vectors(self) -> None:
        vpath = self._vectors_path()
        if vpath is None:
            return
        try:
            with self._lock:
                data = {str(fid): vec for fid, vec in self._vectors.items()}
            vpath.write_text(json.dumps(data), encoding="utf-8")
        except OSError:
            _log.exception("vector sidecar write failed (%s); vectors kept in memory", vpath)

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            facts = data.get("facts")
            if isinstance(facts, list):
                self._facts = [
                    _migrate_fact(f)
                    for f in facts
                    if isinstance(f, dict) and "id" in f and "fact" in f
                ]
            self._next_id = int(data.get("next_id") or (max((f["id"] for f in self._facts), default=0) + 1))
        except (OSError, ValueError):
            _log.exception("fact store unreadable (%s); starting empty", self.path)
            self._facts, self._next_id = [], 1
        vpath = self._vectors_path()
        if vpath is not None and vpath.exists():
            try:
                raw = json.loads(vpath.read_text(encoding="utf-8"))
                ids = {f["id"] for f in self._facts}
                self._vectors = {
                    int(fid): vec for fid, vec in raw.items()
                    if int(fid) in ids and isinstance(vec, list)
                }
            except (OSError, ValueError):
                _log.exception("vector sidecar unreadable (%s); will re-embed", vpath)
                self._vectors = {}


_SOURCES = frozenset({"user", "agent", "host"})


def _repair_json_tail(text: str) -> tuple[str, bool] | None:
    """Referme un JSON tronqué à la FIN. Rend `(réparé, coupé_en_pleine_valeur)`.

    Observé en conditions réelles (Gemini 3.5, août 2026) : la réponse
    d'extraction arrive amputée de son accolade finale —
    ``{"operations": [ {...} ]`` — de façon reproductible et sans rapport avec
    ``max_tokens``. `json.loads` échouait, les opérations étaient abandonnées, et
    **la contradiction était perdue en silence** : la mémoire gardait le fait
    périmé en le servant comme courant. Le pire mode d'échec possible pour une
    mémoire.

    On ne « devine » rien : on parcourt le texte en suivant l'état chaîne /
    échappement, et on ne ferme que les délimiteurs RESTÉS ouverts. Un JSON dont
    les délimiteurs sont incohérents (fermeture qui ne correspond pas) rend
    ``None`` — ce n'est pas une simple troncature, on ne le touche pas.

    Le second membre du tuple dit si la coupe est tombée EN PLEINE VALEUR (dans
    une chaîne) : l'appelant sait alors que le dernier élément est douteux.
    """
    pile: list[str] = []
    dans_chaine = False
    echappe = False
    for caractere in text:
        if dans_chaine:
            if echappe:
                echappe = False
            elif caractere == "\\":
                echappe = True
            elif caractere == '"':
                dans_chaine = False
            continue
        if caractere == '"':
            dans_chaine = True
        elif caractere in "[{":
            pile.append(caractere)
        elif caractere in "]}":
            attendu = "[" if caractere == "]" else "{"
            if pile and pile[-1] == attendu:
                pile.pop()
            else:
                return None  # incohérent : pas une troncature de queue
    if not pile and not dans_chaine:
        return None  # rien à réparer : l'échec de parsing vient d'ailleurs
    repare = text + ('"' if dans_chaine else "")
    for ouvrant in reversed(pile):
        repare += "]" if ouvrant == "[" else "}"
    return repare, dans_chaine


def _migrate_fact(fact: dict[str, Any]) -> dict[str, Any]:
    """Complète un fait de l'ANCIEN format (0.12→0.17) en place (0.18.0).

    Migration en LECTURE, jamais destructive : un fichier écrit par une version
    antérieure — et il en existe en production — se charge tel quel, les champs
    absents prennent des valeurs qui reproduisent exactement l'ancien
    comportement (aucun fait périmé, aucune supersession, provenance inconnue
    déclarée `agent` puisque c'était l'extraction qui écrivait). Le fichier est
    réécrit au format complet à la première sauvegarde suivante.
    """
    fact.setdefault("subject", None)
    fact.setdefault("updated", "")
    source = fact.get("source")
    fact["source"] = source if source in _SOURCES else "agent"
    fact.setdefault("valid_from", fact.get("updated") or "")
    if "invalid_at" not in fact:
        fact["invalid_at"] = None
    if "superseded_by" not in fact:
        fact["superseded_by"] = None
    return fact


def _tokenize(text: str) -> list[str]:
    """Mots d'au moins 2 caractères, minuscules, accents PRÉSERVÉS.

    L'ancien recall faisait `text.lower().split()` : « crêpes, » ne matchait donc
    pas « crêpes », et un seuil à 3 caractères jetait « n° », « TVA », « ok ».
    `\\w` est unicode-aware en Python 3 → « crêpes » et « héberger » survivent.
    """
    return [t for t in re.split(r"\W+", (text or "").lower(), flags=re.UNICODE) if len(t) >= 2]


def _rrf_fuse(*rankings: list[dict[str, Any]], k: int = 60) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion : score = Σ 1/(k + rang).

    On fusionne des RANGS, pas des scores : un cosinus (0→1) et un score BM25
    (non borné) ne sont pas comparables, mais leurs positions le sont. `k=60` est
    la constante usuelle de la littérature ; elle amortit la tête de classement
    pour qu'un seul signal très confiant ne balaie pas l'autre.
    """
    scores: dict[int, float] = {}
    seen: dict[int, dict[str, Any]] = {}
    for ranking in rankings:
        for rank, fact in enumerate(ranking, start=1):
            fid = fact["id"]
            scores[fid] = scores.get(fid, 0.0) + 1.0 / (k + rank)
            seen.setdefault(fid, fact)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [seen[fid] for fid, _ in ordered]


def _cosine(a: list[float], b: list[float]) -> float:
    """Similarité cosinus en pur stdlib (les vecteurs sont courts)."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _prefix_fingerprint(messages: list[Message]) -> str:
    """Empreinte stable d'un préfixe de conversation (rôles + contenus)."""
    hasher = hashlib.sha256()
    for message in messages:
        hasher.update(message.role.encode())
        hasher.update(b"\x01")
        hasher.update((message.content or "").encode("utf-8", "replace"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def _parse_forget_ids(content: str) -> set[int]:
    """Extrait les ids à oublier. Tolérant aux fences ```json.

    Fail-CLOSED : tout ce qui n'est pas un entier clairement listé est ignoré —
    on ne devine JAMAIS une suppression.

    Volontairement PAS de réparation de JSON tronqué ici, contrairement à
    l'extraction (`_parse_operations`) : sur une troncature, `[123]` amputé donne
    `[12]` — un id VALIDE mais faux, donc la suppression d'un fait innocent. Les
    deux chemins n'ont pas le même risque : perdre une opération d'extraction se
    rattrape au tour suivant, détruire la donnée d'un client ne se rattrape pas.
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
    except ValueError:
        _log.warning("forget_matching returned non-JSON; nothing deleted")
        return set()
    ids = data.get("forget") if isinstance(data, dict) else None
    if not isinstance(ids, list):
        return set()
    out: set[int] = set()
    for raw in ids:
        if isinstance(raw, bool):
            continue
        if isinstance(raw, int):
            out.add(raw)
        elif isinstance(raw, str) and raw.strip().isdigit():
            out.add(int(raw.strip()))
    return out


def _parse_operations(content: str) -> list[dict[str, Any]]:
    """Extrait la liste d'opérations de la réponse d'extraction.

    Tolérant : fences ```json éventuelles (Anthropic est en best-effort JSON),
    objet mal formé → liste vide (la compaction du tour est alors un no-op
    factuel, jamais une erreur).

    Répare une TRONCATURE de queue (0.18.0) : constaté en réel, Gemini 3.5 renvoie
    parfois `{"operations": [ {...} ]` sans l'accolade finale, de façon
    reproductible et sans lien avec `max_tokens`. On perdait alors TOUTES les
    opérations du tour — donc la contradiction — et la mémoire continuait à servir
    le fait périmé comme courant. Si la coupe est tombée en pleine chaîne, le
    dernier élément est ÉCARTÉ : un fait au texte amputé serait pire que pas de
    fait.
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    coupe_en_valeur = False
    try:
        data = json.loads(text)
    except ValueError:
        repair = _repair_json_tail(text)
        if repair is None:
            _log.warning("fact extraction returned non-JSON; no operations applied")
            return []
        repare, coupe_en_valeur = repair
        try:
            data = json.loads(repare)
        except ValueError:
            _log.warning("fact extraction returned non-JSON; no operations applied")
            return []
        _log.warning("fact extraction JSON was truncated; tail repaired%s",
                     " (last operation dropped)" if coupe_en_valeur else "")
    operations = data.get("operations") if isinstance(data, dict) else None
    if not isinstance(operations, list):
        return []
    valides = [op for op in operations if isinstance(op, dict)]
    if coupe_en_valeur and valides:
        valides.pop()  # élément probablement incomplet : on ne l'applique pas
    return valides
