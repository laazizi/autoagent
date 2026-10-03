"""Le journal durable : ce qu'un run S'APPRÊTE à faire, et ce qu'il a fait — pour qu'une
coupure brutale ne refasse jamais un effet.

Le trou. Un run reprend d'un instantané pris à la fin de chaque étape. Si le processus meurt
PENDANT une étape — après que le premier outil a envoyé son mail, avant l'instantané — la
reprise repart de l'instantané précédent et REFAIT l'étape : le mail part deux fois. C'est le
défaut que la littérature relève chez les frameworks d'agents : à la reprise, les effets sont
refaits (Khan, arXiv:2608.03836, preprint à auteur unique, août 2026 : cinq frameworks audités,
k processus qui reprennent la même pause déclenchent l'effet k fois).

Ce que fait `Journal` — un fichier JSONL, un seul écrivain :

* **l'intention est écrite et `fsync` AVANT l'effet** (`intent`), le résultat APRÈS
  (`result`). À la reprise : un appel avec résultat n'est jamais relancé (son résultat est
  réinjecté) ; un appel avec intention mais sans résultat a une **issue inconnue** — l'effet
  a peut-être eu lieu — et n'est PAS relancé en silence : `OutcomeUnknown`, une pause humaine,
  sauf pour un outil déclaré `idempotent` ;
* **FAIL-CLOSED** : si l'intention ne peut pas être écrite, l'effet ne part pas (`JournalError`).
  C'est le contrat inverse de la trace (observabilité, fail-open) : un journal qui se tait n'est
  plus un journal. Les deux restent donc deux mécanismes ;
* **chaîne d'empreintes** : chaque enregistrement porte l'empreinte du précédent. Retirer ou
  modifier un enregistrement au milieu du fichier rompt la chaîne (`verify()`,
  `JournalCorrupted`). Ce n'est PAS une signature : qui peut réécrire tout le fichier peut
  recalculer toute la chaîne — pour une preuve opposable, ancre `head()` ailleurs ;
* **un seul écrivain** (verrou du système sur `<fichier>.lock`, libéré à la mort du processus) :
  deux processus qui reprennent le même run ne refont pas chacun l'effet en cours ;
* **la queue abîmée est réparée** : un dernier enregistrement à moitié écrit (coupure pendant
  l'écriture) est retiré à l'ouverture — jamais un enregistrement du milieu.

Ce qu'il contient : les arguments et les résultats COMPLETS des outils, et la conversation —
c'est ce qu'il faut pour reprendre sans refaire. Protège-le comme un instantané (droits du
fichier, durée de conservation) : ce n'est pas un journal à expédier tel quel à un tiers.

    journal = Journal("run.jsonl")
    agent = Agent(provider, journal=journal)
    agent.run("…")
    # après une coupure brutale, dans un nouveau processus :
    agent = Agent(provider, journal=Journal("run.jsonl"))
    agent.resume_from_journal()        # lève `OutcomeUnknown` si un effet a une issue inconnue
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import JournalCorrupted, JournalError, JournalLocked
from .logging import get_logger
from .registry import ToolResult
from .schema import Message, ToolCall

if TYPE_CHECKING:
    from .agent import RunState

__all__ = ["CallView", "Journal", "JournalReport", "RunView", "idempotency_key"]

_log = get_logger("journal")

_GENESIS = "0" * 32
# Statuts de fin après lesquels un run peut être repris (le reste est TERMINÉ).
_REPRENABLES = ("approval_required", "cancelled", "max_steps", "token_budget")

# La clé d'idempotence de l'appel d'outil EN COURS (posée par la boucle autour de chaque
# exécution, comme la porte de décision) : stable d'une reprise à l'autre — c'est ce qui permet
# à un outil de la passer au système externe (paiement, envoi) pour qu'il dédoublonne lui-même.
_CLE: contextvars.ContextVar[str | None] = contextvars.ContextVar("autoagent_idempotency_key", default=None)


def idempotency_key() -> str | None:
    """La clé d'idempotence de l'appel d'outil en cours, ou `None` hors d'un run journalisé.

    Stable : un même appel (même run, même `call.id`) reçoit la même clé avant et après une
    coupure. À transmettre au service externe (en-tête d'idempotence d'une API de paiement…) :
    s'il l'a déjà vue, il ne refait pas l'effet. Sans journal, `None`.
    """
    return _CLE.get()


@contextmanager
def key_scope(key: str | None) -> Iterator[None]:
    """Pose la clé d'idempotence le temps d'UNE exécution d'outil (utilisé par la boucle)."""
    token = _CLE.set(key)
    try:
        yield
    finally:
        _CLE.reset(token)


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(prev: str, record: dict[str, Any]) -> str:
    corps = {k: v for k, v in record.items() if k != "hash"}
    return hashlib.sha256((prev + "\n" + _canon(corps)).encode("utf-8")).hexdigest()[:32]


def _enregistrement_valide(rec: Any, prev: str, rang: int) -> bool:
    return (isinstance(rec, dict) and rec.get("prev") == prev and rec.get("seq") == rang
            and rec.get("hash") == _digest(prev, rec))


def _verrouiller(handle: Any) -> None:
    """Verrou EXCLUSIF non bloquant sur un fichier dédié. Libéré par le système à la mort du
    processus : pas de verrou périmé à nettoyer après une coupure brutale."""
    if sys.platform == "win32":                            # pragma: no cover — selon la plateforme
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:                                                  # pragma: no cover — selon la plateforme
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


@dataclass
class CallView:
    """Ce que le journal sait d'UN appel d'outil, par une machine à états lue dans l'ORDRE :

    `fresh` (rien d'écrit) → `in_doubt` (intention écrite, pas de résultat : l'effet a peut-être
    eu lieu) → `completed` (résultat écrit, ou décidé par l'hôte) ; ou `retry` (l'hôte a dit
    « il n'est pas parti, relance »). Une NOUVELLE intention après un `retry` redevient `in_doubt` :
    une seconde coupure ne relance pas l'appel en silence une troisième fois.
    """

    call: ToolCall | None = None
    intent: dict[str, Any] | None = None
    outcome: dict[str, Any] | None = None       # {"ok", "result", "error"} — format de `ToolResult`
    state: str = "fresh"
    tainted: bool = False                       # le résultat vient de contenu non fiable lu par le programme

    @property
    def completed(self) -> ToolResult | None:
        """Le résultat à réinjecter (écrit par le run, ou décidé par l'hôte), ou `None`."""
        if self.state != "completed" or self.outcome is None:
            return None
        return ToolResult.from_dict(self.outcome)

    @property
    def in_doubt(self) -> bool:
        """Intention écrite, ni résultat ni décision de l'hôte : l'effet a PEUT-ÊTRE eu lieu."""
        return self.state == "in_doubt"


@dataclass
class RunView:
    """Un run reconstitué depuis le journal — de quoi le reprendre."""

    run_id: str
    messages: list[Message] = field(default_factory=list)
    counters: dict[str, Any] = field(default_factory=dict)
    calls: dict[str, CallView] = field(default_factory=dict)
    status: str | None = None          # statut du `run_end`, ou None si le run n'a pas fini

    @property
    def finished(self) -> bool:
        return self.status is not None and self.status not in _REPRENABLES

    def pending_calls(self) -> list[ToolCall]:
        """Les appels du DERNIER message de l'assistant dont les résultats ne sont pas encore
        dans la conversation (l'étape était en cours quand le run s'est arrêté)."""
        if self.messages and self.messages[-1].role == "assistant":
            return list(self.messages[-1].tool_calls or [])
        return []

    def completed(self) -> dict[str, ToolResult]:
        return {cid: r for cid, v in self.calls.items() if (r := v.completed) is not None}

    def bridge_tainted(self) -> set[str]:
        """Les appels terminés dont l'exécution a lu du contenu non fiable (teinte du run)."""
        return {cid for cid, v in self.calls.items() if v.tainted and v.completed is not None}

    def in_doubt(self) -> list[ToolCall]:
        """Les appels en attente dont l'issue est inconnue (ordre de la demande)."""
        return [c for c in self.pending_calls() if (v := self.calls.get(c.id)) is not None and v.in_doubt]

    def to_run_state(self) -> RunState:
        from .agent import RunState
        k = self.counters
        return RunState(
            messages=list(self.messages), step=int(k.get("step", 0)),
            corrections=int(k.get("corrections", 0)), turn_start=int(k.get("turn_start", 0)),
            input_tokens=int(k.get("input_tokens", 0)), output_tokens=int(k.get("output_tokens", 0)),
            cached_tokens=int(k.get("cached_tokens", 0)), have_usage=bool(k.get("have_usage", False)),
            tainted=bool(k.get("tainted", False)),
        )


@dataclass
class JournalReport:
    """Résultat de `Journal.verify()`."""

    ok: bool
    records: int
    head: str
    first_bad: int | None = None
    detail: str = ""


class Journal:
    """Un journal durable, chaîné, à écrivain unique. Voir l'en-tête du module.

    Args:
        path: le fichier JSONL (créé, ainsi que son dossier, au besoin).
        fsync: `True` (défaut) : intention, résultat et instantanés sont forcés sur le disque
            avant que l'on continue. `False` : plus rapide, plus de garantie contre une coupure
            de courant — à réserver aux tests.
        on_boundary: `(nom, infos)` appelé aux frontières de l'écriture — `state:before/after`,
            `intent:after`, `effect:after`, `result:after`, `end:before`. Sert à TUER le processus
            à chaque frontière dans les tests de coupure ; un hôte peut y brancher de la
            télémétrie. Une exception levée ici remonte (c'est voulu, pour les tests).
    """

    def __init__(self, path: str | Path, *, fsync: bool = True,
                 on_boundary: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        self.path = Path(path)
        self._fsync = fsync
        self._on_boundary = on_boundary
        self._lock = threading.RLock()
        self._records: list[dict[str, Any]] = []
        self._prev = _GENESIS
        self._seq = 0
        self._casse = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._verrou = open(str(self.path) + ".lock", "a+b")   # noqa: SIM115 — vit autant que le journal
        try:
            _verrouiller(self._verrou)
        except OSError as exc:
            self._verrou.close()
            raise JournalLocked(f"{self.path} est déjà ouvert en écriture par un autre processus") from exc
        try:
            self._charger()
            self._fh = open(self.path, "ab")                   # noqa: SIM115
        except BaseException:
            self._verrou.close()
            raise

    # ── ouverture : lire, vérifier, réparer la queue ─────────────────────────

    def _charger(self) -> None:
        """Lit le fichier, vérifie la chaîne, retire une queue abîmée.

        Seule la DERNIÈRE ligne peut être abîmée par une coupure (écriture à moitié faite,
        ou page du disque à moitié persistée) : on la retire. Une ligne illisible ou
        incohérente AILLEURS, c'est une modification : `JournalCorrupted`, jamais une réparation
        silencieuse."""
        if not self.path.exists():
            return
        lignes = self.path.read_bytes().split(b"\n")
        queue = lignes.pop()           # b"" si le fichier finit par un saut de ligne ; sinon, ligne incomplète
        a_retirer = bool(queue)
        enregistrements: list[dict[str, Any]] = []
        prev = _GENESIS
        octets_valides = 0
        for rang, ligne in enumerate(lignes):
            try:
                rec: Any = json.loads(ligne)
            except ValueError:
                rec = None
            if not _enregistrement_valide(rec, prev, rang):
                if rang == len(lignes) - 1 and not a_retirer:
                    a_retirer = True   # dernière ligne terminée mais abîmée (coupure de courant)
                    break
                exc = JournalCorrupted(f"journal corrompu à l'enregistrement {rang} : "
                                       "la chaîne d'empreintes ne se vérifie plus")
                exc.seq = rang
                raise exc
            enregistrements.append(rec)
            prev = rec["hash"]
            octets_valides += len(ligne) + 1
        if a_retirer:
            _log.warning("journal %s : dernier enregistrement incomplet retiré (coupure pendant "
                         "l'écriture)", self.path)
            with open(self.path, "r+b") as f:
                f.truncate(octets_valides)
                f.flush()
                os.fsync(f.fileno())
        self._records = enregistrements
        self._prev = prev
        self._seq = len(enregistrements)

    # ── écriture ─────────────────────────────────────────────────────────────

    def boundary(self, nom: str, **infos: Any) -> None:
        """Signale une frontière d'écriture au crochet `on_boundary` (aucun effet sinon)."""
        if self._on_boundary is not None:
            self._on_boundary(nom, infos)

    def append(self, type_: str, run: str | None, data: dict[str, Any], *, sync: bool = True) -> dict[str, Any]:
        """Ajoute UN enregistrement chaîné. Rend l'enregistrement écrit. Lève `JournalError`
        si l'écriture échoue — et refuse toute écriture suivante : une ligne à moitié écrite au
        milieu du fichier romprait la chaîne."""
        with self._lock:
            if self._casse:
                raise JournalError("le journal est dans un état inconnu après une écriture échouée : "
                                   "rouvre-le (la queue incomplète sera réparée)")
            rec: dict[str, Any] = {"seq": self._seq, "ts": time.time(), "run": run, "type": type_,
                                   "data": data, "prev": self._prev}
            rec["hash"] = _digest(self._prev, rec)
            try:
                self._fh.write((_canon(rec) + "\n").encode("utf-8"))
                self._fh.flush()
                if sync and self._fsync:
                    os.fsync(self._fh.fileno())
            except (OSError, ValueError) as exc:
                self._casse = True
                raise JournalError(f"écriture du journal impossible : {exc}") from exc
            self._records.append(rec)
            self._prev = rec["hash"]
            self._seq += 1
            return rec

    def begin_run(self, *, resumes: str | None = None) -> str:
        """Ouvre un run (ou en reprend un : `resumes=<run_id>` écrit un marqueur de reprise)."""
        run = resumes or uuid.uuid4().hex[:16]
        self.append("resume" if resumes else "run_start", run, {})
        return run

    def record_state(self, run: str, counters: dict[str, Any], delta: list[dict[str, Any]]) -> None:
        """Écrit le DELTA de conversation depuis le dernier état + les compteurs, `fsync`.
        Un delta (pas la conversation entière) : le fichier grossit linéairement."""
        self.boundary("state:before", run=run)
        self.append("state", run, {"counters": counters, "delta": delta})
        self.boundary("state:after", run=run)

    def record_intent(self, run: str, call: ToolCall, *, step: int, idempotent: bool) -> str:
        """L'INTENTION, écrite et forcée sur le disque AVANT que l'outil ne tourne. Rend la clé
        d'idempotence de l'appel. Fail-closed : si ceci échoue, l'effet ne part pas."""
        cle = self.key_for(run, call.id)
        self.append("intent", run, {"call": call.to_dict(), "key": cle, "step": step, "idempotent": idempotent})
        self.boundary("intent:after", run=run, call_id=call.id, name=call.name)
        return cle

    def record_result(self, run: str, call: ToolCall, result: ToolResult, *, tainted: bool = False) -> None:
        """Le RÉSULTAT, après l'effet. L'écart entre `intent` et `result` est la fenêtre où une
        coupure laisse une issue inconnue. `tainted` : l'outil a lu du contenu non fiable par une
        fonction de l'hôte pendant son exécution (D1) — la teinte doit survivre à la coupure."""
        self.boundary("effect:after", run=run, call_id=call.id, name=call.name)
        self.append("result", run, {"call_id": call.id, "result": result.to_dict(), "bridge_tainted": tainted})
        self.boundary("result:after", run=run, call_id=call.id, name=call.name)

    def record_end(self, run: str, status: str) -> None:
        self.boundary("end:before", run=run, status=status)
        self.append("run_end", run, {"status": status})

    def resolve(self, call_id: str, *, ok: bool = True, result: Any = None, error: str | None = None,
                retry: bool = False, by: str = "host", run: str | None = None) -> None:
        """Tranche UN appel à l'issue inconnue — décision de l'HÔTE, écrite dans la chaîne.

        `ok`/`result`/`error` : ce que l'appel a réellement donné (il est parti, voici sa
        réponse) — la reprise réinjecte ce résultat sans rien relancer. `retry=True` : il n'est
        pas parti, relance-le (avec la MÊME clé d'idempotence)."""
        run = run or self._run_of(call_id)
        self.append("resolved", run, {"call_id": call_id, "ok": ok, "result": result, "error": error,
                                      "retry": retry, "by": by})

    def _run_of(self, call_id: str) -> str:
        for rec in reversed(self.records()):
            if rec["type"] == "intent" and rec["data"]["call"].get("id") == call_id:
                return str(rec["run"])
        raise JournalError(f"aucune intention enregistrée pour l'appel {call_id!r}")

    @staticmethod
    def key_for(run: str, call_id: str) -> str:
        """La clé d'idempotence d'un appel : stable avant et après une coupure."""
        return hashlib.sha256(f"{run}:{call_id}".encode()).hexdigest()[:32]

    # ── lecture ──────────────────────────────────────────────────────────────

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records)

    def head(self) -> str:
        """L'empreinte du dernier enregistrement — à ancrer ailleurs pour une preuve opposable."""
        return self._prev

    def verify(self) -> JournalReport:
        """Rejoue la chaîne depuis le début : `ok=False` et `first_bad` si elle est rompue."""
        prev = _GENESIS
        enregistrements = self.records()
        for rang, rec in enumerate(enregistrements):
            if not _enregistrement_valide(rec, prev, rang):
                return JournalReport(False, len(enregistrements), self.head(), rang, "chaîne rompue")
            prev = rec["hash"]
        return JournalReport(True, len(enregistrements), prev)

    def run_view(self, run_id: str) -> RunView:
        """Reconstitue un run : conversation, compteurs, ce que chaque appel a donné."""
        vue = RunView(run_id=run_id)
        for rec in self.records():
            if rec["run"] != run_id:
                continue
            t, d = rec["type"], rec["data"]
            if t == "state":
                vue.counters = dict(d["counters"])
                nouveaux = [Message.from_dict(m) for m in d["delta"]]
                vue.messages.extend(nouveaux)
                for m in nouveaux:
                    for call in m.tool_calls or []:
                        vue.calls.setdefault(call.id, CallView()).call = call
            elif t == "intent":
                call = ToolCall.from_dict(d["call"])
                v = vue.calls.setdefault(call.id, CallView())
                v.call, v.intent, v.outcome, v.state = call, d, None, "in_doubt"
            elif t == "result":
                v = vue.calls.setdefault(d["call_id"], CallView())
                v.outcome, v.state, v.tainted = d["result"], "completed", bool(d.get("bridge_tainted"))
            elif t == "resolved":
                v = vue.calls.setdefault(d["call_id"], CallView())
                if d.get("retry"):
                    v.outcome, v.state = None, "retry"
                else:
                    v.outcome = {"ok": bool(d.get("ok", True)), "result": d.get("result"), "error": d.get("error")}
                    v.state = "completed"
            elif t == "run_end":
                vue.status = d["status"]
            elif t in ("run_start", "resume"):
                vue.status = None                       # un run repris n'est plus terminé
        return vue

    def run_ids(self) -> list[str]:
        vus: list[str] = []
        for rec in self.records():
            if rec["run"] and rec["run"] not in vus:
                vus.append(str(rec["run"]))
        return vus

    def last_resumable_run(self) -> RunView | None:
        """Le dernier run à reprendre : mort sans `run_end` (coupure brutale), ou arrêté sur une
        pause / une borne reprenable. `None` s'il n'y en a pas."""
        for run_id in reversed(self.run_ids()):
            vue = self.run_view(run_id)
            if not vue.finished and vue.messages:
                return vue
        return None

    # ── fermeture ────────────────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            finally:
                self._verrou.close()                # libère le verrou

    def __enter__(self) -> Journal:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
