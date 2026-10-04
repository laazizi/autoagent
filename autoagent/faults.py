"""Un banc de PANNES fournisseur : un vrai serveur HTTP local qui tombe en panne sur commande (0.24.0).

Un agent vocal vit ou meurt sur ce que fait la bibliothèque quand le fournisseur flanche : un 529 « surchargé »,
une erreur annoncée EN COURS de flux, une coupure en pleine phrase, un silence. La 0.23.1 a corrigé des cas où
un flux coupé finissait en « succès » tronqué — mais chaque correctif avait été prouvé sur des réponses SIMULÉES
qu'on injecte à la place de `post_sse`. Ce module prouve la chose avec de vrais sockets : `FaultServer` est un
serveur HTTP (stdlib, `127.0.0.1`, port libre) qui parle le format de fil d'OpenAI (et des compatibles : DeepSeek,
OpenRouter…), d'Anthropic et de Gemini, et dont on script les pannes ; `run_fault_bench` y branche les VRAIS
fournisseurs de la bibliothèque et juge — en code — ce qui en sort.

**Le contrat jugé** : face à une panne, l'appel doit se terminer par une réponse COMPLÈTE, ou par une erreur
TYPÉE (`ProviderError`, avec `retryable` qui dit si cela vaut la peine de réessayer) — jamais un « succès » tronqué
(le texte coupé rendu comme une réponse), jamais une exception brute (`OSError`, `IncompleteRead`…) que l'hôte ne
sait pas attraper, jamais un appel qui ne revient pas. Cinq issues : `complete`, `typed_error`, `truncated_success`,
`untyped_error`, `hang`. Les trois dernières sont des DÉFAUTS ; chaque cas dit quelles issues il accepte
(« 529 puis succès » doit finir `complete` : les relances absorbent la panne ; « 400 » doit finir `typed_error` en
UNE requête : on ne réessaie pas une faute de l'appelant).

**Ce que ça ne fait pas.** Un serveur local n'est pas un fournisseur : pas de TLS, pas de latence réseau, pas de
quotas ; les formats de fil sont reproduits d'après les adaptateurs de la bibliothèque et les documentations, pas
d'après un traçage du vrai service. Il prouve que NOTRE code traite proprement ces pannes-là, pas que le vrai service
ne s'y prend pas autrement.

Exemple::

    from autoagent.faults import run_fault_bench

    rapport = run_fault_bench()                 # tous les fournisseurs, toutes les pannes, hors réseau
    print(rapport.summary())
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .errors import AutoAgentError, ProviderError
from .http import close_connections
from .logging import get_logger
from .providers.base import LLMProvider
from .schema import LLMRequest, Message, ModelConfig

__all__ = [
    "FAULT_CASES",
    "FAULTS",
    "FaultCase",
    "FaultReport",
    "FaultRow",
    "FaultServer",
    "default_provider",
    "run_fault_bench",
]

_log = get_logger("faults")


def _verifier(pannes: Sequence[str]) -> list[str]:
    """Une panne inconnue est une erreur du BANC, dite tout de suite — pas un serveur qui plante en silence."""
    out = []
    for p in pannes:
        statut = p.startswith("http_") and p[5:].isdigit() and 400 <= int(p[5:]) <= 599
        if statut or (p in FAULTS and p != "http_NNN"):
            out.append(p)
        else:
            raise ValueError(f"panne inconnue : {p!r} (voir autoagent.faults.FAULTS)")
    return out


class _Serveur(ThreadingHTTPServer):
    """Un client qui part (délai, coupure voulue) n'est pas une erreur du serveur : rien à imprimer."""

    def handle_error(self, request: Any, client_address: Any) -> None:
        return

TEXTE = "Bonjour, ceci est une réponse complète."
_MORCEAUX = ("Bonjour, ", "ceci est ", "une réponse ", "complète.")

# Les pannes scriptables. « http_NNN » : n'importe quel code HTTP d'erreur.
FAULTS = {
    "ok": "une réponse complète et valide",
    "http_NNN": "un statut HTTP d'erreur (429, 500, 503, 529, 400, 401…) ; les 429 / 5xx portent `retry-after-ms: 20`",
    "body_error": "HTTP 200 dont le corps est une erreur JSON (certaines passerelles font cela) — hors flux",
    "garbage": "HTTP 200 dont le corps n'est pas du JSON valide",
    "silence": "la requête est acceptée, puis plus rien : le client doit sortir sur son délai",
    "sse_error": "un flux qui annonce une erreur EN COURS de route, après quelques morceaux de texte",
    "cut_clean": "un flux coupé PROPREMENT (fin de réponse HTTP valide) avant tout marqueur de fin — un proxy qui ferme",
    "cut_reset": "un flux dont la connexion est coupée net au milieu d'un morceau (le serveur la ferme : IncompleteRead)",
    "stall_mid_stream": "un flux qui se fige en plein texte : plus aucun octet, connexion ouverte",
    "empty_stream": "un flux (HTTP 200) qui se termine sans AUCUN événement",
}


# ── Les formats de fil : ce que dit un fournisseur, morceau par morceau ──────────────────────────────

def _sse(data: dict[str, Any], event: str | None = None) -> bytes:
    tete = f"event: {event}\n" if event else ""
    return f"{tete}data: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def _flux_ok(wire: str) -> Iterator[bytes]:
    """Les événements d'un flux COMPLET et valide, dans l'ordre où le fournisseur les envoie."""
    if wire == "openai":
        for m in _MORCEAUX:
            yield _sse({"model": "fake", "choices": [{"index": 0, "delta": {"content": m}}]})
        yield _sse({"model": "fake", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14}})
        yield b"data: [DONE]\n\n"
    elif wire == "anthropic":
        yield _sse({"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant",
                    "model": "fake", "content": [], "usage": {"input_tokens": 5, "output_tokens": 1}}}, "message_start")
        yield _sse({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                   "content_block_start")
        for m in _MORCEAUX:
            yield _sse({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": m}},
                       "content_block_delta")
        yield _sse({"type": "content_block_stop", "index": 0}, "content_block_stop")
        yield _sse({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 9}}, "message_delta")
        yield _sse({"type": "message_stop"}, "message_stop")
    else:                                                      # gemini
        for m in _MORCEAUX[:-1]:
            yield _sse({"candidates": [{"content": {"role": "model", "parts": [{"text": m}]}, "index": 0}]})
        yield _sse({"candidates": [{"content": {"role": "model", "parts": [{"text": _MORCEAUX[-1]}]},
                                    "finishReason": "STOP", "index": 0}],
                    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 9, "totalTokenCount": 14}})


def _evenement_d_erreur(wire: str) -> bytes:
    if wire == "openai":
        return _sse({"error": {"message": "The server is overloaded", "type": "server_error", "code": 503}})
    if wire == "anthropic":
        return _sse({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}, "error")
    return _sse({"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}})


def _corps_ok(wire: str) -> dict[str, Any]:
    if wire == "openai":
        return {"id": "x", "model": "fake", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": TEXTE}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14}}
    if wire == "anthropic":
        return {"id": "msg_1", "type": "message", "role": "assistant", "model": "fake", "stop_reason": "end_turn",
                "content": [{"type": "text", "text": TEXTE}], "usage": {"input_tokens": 5, "output_tokens": 9}}
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": TEXTE}]}, "finishReason": "STOP",
                            "index": 0}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 9, "totalTokenCount": 14}}


def _corps_d_erreur(wire: str) -> dict[str, Any]:
    if wire == "anthropic":
        return {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    return {"error": {"message": "The server is overloaded", "code": 503, "status": "UNAVAILABLE"}}


# ── Le serveur ───────────────────────────────────────────────────────────────

@dataclass
class RequestRecord:
    """Une requête REÇUE par le serveur : combien de fois le client a réessayé, et à quel rythme."""

    wire: str
    path: str
    stream: bool
    fault: str
    at: float
    body: dict[str, Any] = field(default_factory=dict)


class FaultServer:
    """Un faux fournisseur LLM local, dont on script les pannes — un vrai serveur HTTP/1.1 (stdlib).

    `script` : une panne par REQUÊTE, dans l'ordre (voir `FAULTS`) ; épuisé, il répond `ok`. Un même serveur parle
    OpenAI-compatible (`…/chat/completions`), Anthropic (`…/v1/messages`) et Gemini
    (`…/models/<m>:generateContent` et `:streamGenerateContent`) : le format est déduit du chemin. `url` est le
    `base_url` à donner au fournisseur ; `requests` garde ce qui a été reçu.

    À utiliser en `with` : le serveur démarre dans un fil, et sa sortie libère les requêtes figées
    (`silence`, `stall_mid_stream`) avant d'arrêter."""

    def __init__(self, script: Sequence[str] = (), *, retry_after_ms: int = 20, stall_seconds: float = 30.0) -> None:
        self._script = _verifier(script)
        self._verrou = threading.Lock()
        self.requests: list[RequestRecord] = []
        self.retry_after_ms = retry_after_ms
        self.stall_seconds = stall_seconds
        self._liberer = threading.Event()
        serveur = self

        class Gestionnaire(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:       # le serveur ne parle pas : le rapport parle
                return

            def do_POST(self) -> None:                       # nom imposé par http.server
                serveur._servir(self)

        self._httpd = _Serveur(("127.0.0.1", 0), Gestionnaire)
        self._httpd.daemon_threads = True
        self._fil: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def set_script(self, *pannes: str) -> None:
        with self._verrou:
            self._script = _verifier(pannes)

    def __enter__(self) -> FaultServer:
        self._fil = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self._fil.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._liberer.set()
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._fil is not None:
            self._fil.join(timeout=2)

    # -- le service d'UNE requête --------------------------------------------------------------

    def _prochaine_panne(self) -> str:
        with self._verrou:
            return self._script.pop(0) if self._script else "ok"

    def _servir(self, h: BaseHTTPRequestHandler) -> None:
        taille = int(h.headers.get("Content-Length") or 0)
        brut = h.rfile.read(taille) if taille else b""
        try:
            corps = json.loads(brut or b"{}")
        except ValueError:
            corps = {}
        chemin = h.path
        wire = "anthropic" if "/v1/messages" in chemin else "gemini" if ":generateContent" in chemin \
            or ":streamGenerateContent" in chemin else "openai"
        flux = bool(corps.get("stream")) or ":streamGenerateContent" in chemin
        panne = self._prochaine_panne()
        with self._verrou:
            self.requests.append(RequestRecord(wire, chemin, flux, panne, time.perf_counter(), corps))
        try:
            self._repondre(h, wire, flux, panne)
        except (BrokenPipeError, ConnectionError, OSError):
            h.close_connection = True                       # le client est parti (délai) : rien à dire de plus

    def _repondre(self, h: BaseHTTPRequestHandler, wire: str, flux: bool, panne: str) -> None:
        if panne.startswith("http_"):
            self._erreur_http(h, wire, int(panne[5:]))
        elif panne == "body_error":
            self._json(h, 200, _corps_d_erreur(wire))
        elif panne == "garbage":
            self._brut(h, 200, b"<html>502 Bad Gateway \x00\xff not json</html>", "application/json")
        elif panne == "silence":
            self._liberer.wait(self.stall_seconds)
            h.close_connection = True
        elif not flux:                                       # une panne de flux sur un appel sans flux : une réponse valide
            self._json(h, 200, _corps_ok(wire))
        else:
            self._flux(h, wire, panne)

    # -- les réponses -----------------------------------------------------------------------------

    def _brut(self, h: BaseHTTPRequestHandler, code: int, corps: bytes, type_: str,
              extra: dict[str, str] | None = None) -> None:
        h.send_response(code)
        h.send_header("Content-Type", type_)
        h.send_header("Content-Length", str(len(corps)))
        for k, v in (extra or {}).items():
            h.send_header(k, v)
        h.end_headers()
        h.wfile.write(corps)
        h.wfile.flush()

    def _json(self, h: BaseHTTPRequestHandler, code: int, donnee: dict[str, Any]) -> None:
        self._brut(h, code, json.dumps(donnee, ensure_ascii=False).encode(), "application/json")

    def _erreur_http(self, h: BaseHTTPRequestHandler, wire: str, code: int) -> None:
        extra = {"retry-after-ms": str(self.retry_after_ms)} if code == 429 or code >= 500 else {}
        corps = _corps_d_erreur(wire)
        if wire != "anthropic":                  # Anthropic n'inscrit pas le code dans le corps
            corps["error"]["code"] = code
        self._brut(h, code, json.dumps(corps).encode(), "application/json", extra)

    def _entete_flux(self, h: BaseHTTPRequestHandler) -> None:
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Cache-Control", "no-cache")
        h.send_header("Transfer-Encoding", "chunked")
        h.end_headers()

    @staticmethod
    def _morceau(h: BaseHTTPRequestHandler, donnee: bytes) -> None:
        h.wfile.write(f"{len(donnee):x}\r\n".encode() + donnee + b"\r\n")
        h.wfile.flush()

    def _fin_propre(self, h: BaseHTTPRequestHandler) -> None:
        h.wfile.write(b"0\r\n\r\n")
        h.wfile.flush()

    def _flux(self, h: BaseHTTPRequestHandler, wire: str, panne: str) -> None:
        self._entete_flux(h)
        evenements = list(_flux_ok(wire))
        if panne == "ok":
            for e in evenements:
                self._morceau(h, e)
            self._fin_propre(h)
        elif panne == "empty_stream":
            self._fin_propre(h)
        elif panne in ("sse_error", "cut_clean", "cut_reset", "stall_mid_stream"):
            # Quelques morceaux de texte, puis la panne : de quoi avoir DÉJÀ émis du texte avant d'échouer.
            avant = 3 if wire == "anthropic" else 2
            for e in evenements[:avant + (1 if wire == "anthropic" else 0)]:
                self._morceau(h, e)
            if panne == "sse_error":
                self._morceau(h, _evenement_d_erreur(wire))
                self._fin_propre(h)
            elif panne == "cut_clean":
                self._fin_propre(h)
            elif panne == "cut_reset":
                h.wfile.write(b"ffff\r\ndata: {\"partiel\":")        # un morceau annoncé de 65 535 octets, 18 envoyés
                h.wfile.flush()
                h.close_connection = True
                try:
                    h.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            else:                                                     # stall_mid_stream
                self._liberer.wait(self.stall_seconds)
                h.close_connection = True
        else:
            raise ValueError(f"panne inconnue : {panne!r}")


# ── Le banc ──────────────────────────────────────────────────────────────────

OK_ISSUES = ("complete", "typed_error")


@dataclass(frozen=True)
class FaultCase:
    """Une panne scriptée et ce que le contrat en accepte. `script` : une panne PAR REQUÊTE (les relances du client
    consomment les suivantes). `accept` : les issues acceptables ; `max_requests` : au plus N requêtes reçues
    (une faute de l'appelant ne se réessaie pas)."""

    name: str
    script: tuple[str, ...]
    stream: bool
    accept: tuple[str, ...] = OK_ISSUES
    max_requests: int | None = None
    note: str = ""


FAULT_CASES: tuple[FaultCase, ...] = (
    FaultCase("temoin", ("ok",), False, ("complete",), 1, "aucune panne : le banc doit voir une réponse complète"),
    FaultCase("temoin_flux", ("ok",), True, ("complete",), 1, "idem, en flux"),
    FaultCase("surcharge_puis_succes", ("http_529", "ok"), False, ("complete",), 2,
              "529 « surchargé » puis succès : les relances absorbent la panne"),
    FaultCase("limite_de_debit_puis_succes_flux", ("http_429", "ok"), True, ("complete",), 2,
              "429 puis succès, en flux : la relance a lieu avant le premier octet"),
    FaultCase("erreur_serveur_permanente", ("http_500", "http_500", "http_500"), False, OK_ISSUES, 3,
              "500 sans fin : relances épuisées, erreur typée et réessayable"),
    FaultCase("faute_de_l_appelant", ("http_400", "ok"), False, ("typed_error",), 1,
              "400 : on ne réessaie PAS une requête invalide"),
    FaultCase("cle_refusee", ("http_401", "ok"), True, ("typed_error",), 1, "401 : idem, en flux"),
    FaultCase("corps_d_erreur_en_200", ("body_error",), False, ("typed_error",), None,
              "HTTP 200 dont le corps est une erreur : jamais une réponse vide « réussie »"),
    FaultCase("corps_illisible", ("garbage",), False, ("typed_error",), None, "200 avec un corps qui n'est pas du JSON"),
    FaultCase("erreur_annoncee_en_flux", ("sse_error",), True, ("typed_error",), 1,
              "l'erreur arrive APRÈS du texte : le texte coupé ne doit pas passer pour la réponse"),
    FaultCase("flux_coupe_proprement", ("cut_clean",), True, ("typed_error",), 1,
              "fin de réponse HTTP valide sans marqueur de fin : un texte tronqué n'est pas un succès"),
    FaultCase("flux_coupe_net", ("cut_reset",), True, ("typed_error",), 1,
              "connexion coupée en plein morceau : une erreur TYPÉE, pas un IncompleteRead brut"),
    FaultCase("flux_fige", ("stall_mid_stream",), True, ("typed_error",), 1,
              "plus aucun octet en plein texte : l'appel sort sur son délai, typé"),
    FaultCase("flux_vide", ("empty_stream",), True, ("typed_error",), 1, "200 puis rien : pas une réponse vide « réussie »"),
    FaultCase("silence", ("silence", "silence", "silence"), False, ("typed_error",), 3,
              "requête acceptée puis silence : l'appel sort sur son délai, typé"),
)


def default_provider(wire: str, base_url: str, timeout: float) -> LLMProvider:
    """Le fournisseur STOCK de la bibliothèque pour ce format de fil, pointé sur `base_url`."""
    from .providers import create_provider

    nom = {"openai": "openai", "anthropic": "anthropic", "gemini": "gemini"}[wire]
    return create_provider(ModelConfig(provider=nom, model="fake", base_url=base_url, api_key="cle-de-test",
                                       timeout=timeout))


@dataclass
class FaultRow:
    wire: str
    case: str
    stream: bool
    outcome: str
    accepted: bool
    seconds: float
    requests: int
    detail: str = ""


@dataclass
class FaultReport:
    rows: list[FaultRow]

    @property
    def failures(self) -> list[FaultRow]:
        return [r for r in self.rows if not r.accepted]

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        lignes = [f"Banc de pannes fournisseur : {len(self.rows)} cas, {len(self.failures)} défaut(s)."]
        largeur = max([len(r.case) for r in self.rows] + [8])
        lignes.append(f"{'panne':<{largeur}}  fil        résultat            req.  durée   verdict")
        for r in self.rows:
            lignes.append(f"{r.case:<{largeur}}  {r.wire:<9}  {r.outcome:<18}  {r.requests:>4}  {r.seconds:>5.2f}s  "
                          f"{'ok' if r.accepted else 'DÉFAUT'}" + ("" if r.accepted or not r.detail else f" — {r.detail[:90]}"))
        lignes.append("→ " + ("aucun défaut" if self.ok else f"{len(self.failures)} défaut(s) (lignes DÉFAUT : texte tronqué "
                                                                 "rendu comme une réponse, exception brute, appel qui ne revient "
                                                                 "pas, issue non acceptée ou aucune requête reçue)"))
        lignes.append("Lecture : un serveur LOCAL (pas de TLS, pas de latence réseau) ; formats de fil reproduits d'après "
                      "les adaptateurs et les documentations, pas d'un traçage du vrai service. Le juge vérifie le TYPE de "
                      "l'erreur ; `retryable` est rapporté dans le détail de chaque ligne, il n'est pas jugé.")
        return "\n".join(lignes)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "rows": [vars(r).copy() for r in self.rows]}


def _appeler(provider: LLMProvider, stream: bool) -> tuple[str, str]:
    """Un appel au fournisseur : (issue, détail). Le contrat — complet, ou erreur TYPÉE."""
    requete = LLMRequest(messages=[Message(role="user", content="Dis bonjour.")])
    try:
        if stream:
            texte, final = "", None
            for morceau in provider.stream(requete):
                if morceau.type == "text":
                    texte += morceau.text or ""
                elif morceau.type == "final":
                    final = morceau.response
            if final is None:
                return "truncated_success", "le flux s'est terminé sans réponse finale"
            contenu = final.content
        else:
            contenu = provider.complete(requete).content
    except ProviderError as exc:
        return "typed_error", f"ProviderError(status={exc.status_code}, retryable={exc.retryable}): {exc}"[:200]
    except AutoAgentError as exc:                                                   # le contrat dit `ProviderError`, pas n'importe quoi
        return "untyped_error", f"{type(exc).__name__} (pas une ProviderError) : {exc}"[:200]
    except Exception as exc:                                                        # une exception BRUTE : un défaut
        return "untyped_error", f"{type(exc).__name__}: {exc}"[:200]
    if contenu == TEXTE:
        return "complete", ""
    return "truncated_success", f"rendu comme une réponse : {contenu!r}"[:200]


def _appeler_dans(provider: LLMProvider, stream: bool, sortie: list[tuple[str, str]]) -> None:
    try:
        sortie.append(_appeler(provider, stream))
    except BaseException as exc:                # un `SystemExit` dans le fil n'est pas un « hang » : c'est une exception brute
        sortie.append(("untyped_error", f"{type(exc).__name__} (BaseException) : {exc}"[:200]))
    finally:
        close_connections()        # le pool de connexions est PAR FIL : celui de ce fil ne survit pas au banc, on le ferme


def run_fault_bench(
    *,
    wires: Sequence[str] = ("openai", "anthropic", "gemini"),
    cases: Sequence[FaultCase] = FAULT_CASES,
    timeout: float = 0.6,
    provider_factory: Callable[[str, str, float], LLMProvider] = default_provider,
    deadline: float | None = None,
    on_row: Callable[[FaultRow], None] | None = None,
    quiet: bool = True,
) -> FaultReport:
    """Passe chaque panne devant chaque fournisseur et juge l'issue.

    `provider_factory(wire, base_url, timeout)` : par défaut le fournisseur STOCK ; donne la tienne pour mesurer TA
    configuration (un fournisseur enveloppé, un délai, des relances). `timeout` : le délai réseau du fournisseur —
    court, pour que `silence` et `flux_fige` sortent vite. `deadline` : au-delà, l'appel est déclaré `hang`
    (par défaut : de quoi laisser les relances sortir sur leur délai). `quiet` : les relances de la bibliothèque
    journalisent des WARNING — attendus ici, et bruyants : on les baisse le temps du banc (puis on remet)."""
    journal = logging.getLogger("autoagent")
    niveau = journal.level
    if quiet:
        journal.setLevel(logging.ERROR)
    try:
        return _executer_le_banc(wires, cases, timeout, provider_factory, deadline, on_row)
    finally:
        if quiet:
            journal.setLevel(niveau)


def _executer_le_banc(
    wires: Sequence[str],
    cases: Sequence[FaultCase],
    timeout: float,
    provider_factory: Callable[[str, str, float], LLMProvider],
    deadline: float | None,
    on_row: Callable[[FaultRow], None] | None,
) -> FaultReport:
    limite = deadline if deadline is not None else max(10.0, timeout * 12 + 6)
    lignes: list[FaultRow] = []
    for wire in wires:
        if wire not in ("openai", "anthropic", "gemini"):
            raise ValueError(f"format de fil inconnu : {wire!r}")
        for cas in cases:
            with FaultServer(cas.script, stall_seconds=limite + 5) as serveur:
                provider = provider_factory(wire, serveur.url, timeout)
                resultat: list[tuple[str, str]] = []
                debut = time.perf_counter()
                fil = threading.Thread(target=_appeler_dans, args=(provider, cas.stream, resultat), daemon=True)
                fil.start()
                fil.join(limite)
                duree = time.perf_counter() - debut
                issue, detail = resultat[0] if resultat else ("hang", f"aucune issue après {limite:.0f} s")
                requetes = len(serveur.requests)
            accepte = issue in cas.accept and (cas.max_requests is None or requetes <= cas.max_requests)
            if requetes == 0:
                # Un fournisseur qui échoue AVANT tout réseau passerait sinon les 33 cas qui acceptent `typed_error`.
                accepte = False
                detail = "aucune requête reçue : le fournisseur a échoué avant tout réseau, rien n'a été mesuré" \
                         + (f" ({detail})" if detail else "")
            elif issue in cas.accept and not accepte:
                detail = f"{requetes} requêtes reçues, au plus {cas.max_requests} attendues" + (f" ({detail})" if detail else "")
            elif not accepte and not detail:
                detail = f"issue inattendue : {issue} (acceptées : {', '.join(cas.accept)})"
            ligne = FaultRow(wire, cas.name, cas.stream, issue, accepte, duree, requetes, detail)
            lignes.append(ligne)
            if on_row is not None:
                try:
                    on_row(ligne)
                except Exception:
                    _log.exception("on_row callback failed")                    # fail-open
    return FaultReport(rows=lignes)
