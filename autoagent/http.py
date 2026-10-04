from __future__ import annotations

import email.utils
import http.client
import json
import random
import ssl
import threading
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

from .errors import ProviderError
from .logging import get_logger

__all__ = ["post_json", "post_sse", "close_connections", "is_retryable_status"]

_log = get_logger("http")

_DEFAULT_RETRIES = 2  # total attempts = retries + 1

# HTTP statuses worth retrying: timeouts, conflicts, rate limits and transient
# upstream failures. Other 4xx are caller errors — retrying them only wastes
# time and quota.
#
# 0.23.1 : la règle est celle du SDK officiel d'Anthropic (code relu le 3 oct.
# 2026) — 408, 409, 429 et TOUT 5xx, donc le 529 « overloaded » qui est
# précisément ce qu'Anthropic renvoie quand elle est saturée (la 0.23.0 ne
# relançait que 429/500/502/503/504, et le 529 échouait tout de suite). Seuls
# 501 (non implémenté) et 505 (version HTTP) sont permanents : inutile de les
# rejouer.
_PERMANENT_5XX = frozenset({501, 505})


def is_retryable_status(code: int) -> bool:
    """Ce statut HTTP est-il transitoire (donc à réessayer) ? Partagé avec les fournisseurs,
    pour qu'une erreur reçue EN COURS de flux soit classée comme une erreur HTTP."""
    return code in (408, 409, 429) or (500 <= code <= 599 and code not in _PERMANENT_5XX)

# ── Connexions persistantes (0.21.0) ─────────────────────────────────────────
#
# Jusqu'ici chaque appel passait par `urllib.request.urlopen`, qui ouvre une
# connexion NEUVE : poignée de main TCP + TLS à chaque appel LLM. Mesuré contre
# l'hôte Gemini : ~250 ms la première fois, ~100 ms ensuite, contre 16-40 ms
# sur une connexion réutilisée — soit ~120 ms de surcoût par appel, une seconde
# par run de huit étapes, avant même que le modèle ait commencé à répondre.
#
# On garde donc UNE connexion par (thread, hôte) et on la réutilise. Par thread
# parce que `http.client.HTTPSConnection` n'est pas partageable entre threads
# (parallel_tool_calls, delegate_to, exécution anticipée) ; par hôte parce que
# c'est l'unité de la poignée de main. Une connexion qui casse est jetée et
# recréée — le serveur peut fermer une connexion inactive, c'est prévu : la
# première tentative sur une connexion morte compte comme une erreur transitoire
# et se réessaie sur une connexion neuve. Zéro dépendance : c'est la stdlib.
_local = threading.local()
_ssl_context = ssl.create_default_context()


def _connection(scheme: str, host: str, port: int | None, timeout: float) -> http.client.HTTPConnection:
    pool: dict[tuple[str, str, int | None], http.client.HTTPConnection] = getattr(_local, "pool", None) or {}
    _local.pool = pool
    conn = pool.get((scheme, host, port))
    if conn is None:
        # `http://` reste accepté : Ollama, vLLM, LM Studio écoutent en clair en
        # local, et la version urllib les servait déjà.
        if scheme == "https":
            conn = http.client.HTTPSConnection(host, port, timeout=timeout, context=_ssl_context)
        else:
            conn = http.client.HTTPConnection(host, port, timeout=timeout)
        pool[(scheme, host, port)] = conn
    else:
        conn.timeout = timeout          # l'appelant peut changer de timeout
    return conn


def _discard(scheme: str, host: str, port: int | None) -> None:
    pool = getattr(_local, "pool", None) or {}
    conn = pool.pop((scheme, host, port), None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


def close_connections() -> None:
    """Ferme les connexions persistantes du THREAD courant (tests, arrêt propre)."""
    pool = getattr(_local, "pool", None) or {}
    for conn in pool.values():
        try:
            conn.close()
        except Exception:
            pass
    _local.pool = {}


class _HTTPStatusError(Exception):
    """Réponse HTTP non-2xx, avec le corps lu — l'équivalent de `urllib.error.HTTPError`."""

    def __init__(self, code: int, detail: str, headers: dict[str, str]) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code
        self.detail = detail
        self.headers = headers


def _is_transient(exc: BaseException) -> bool:
    """Coupures de connexion, délais dépassés, connexion fermée par le serveur —
    ça se réessaie. Une résolution DNS impossible ou un hôte inexistant, non :
    on garde l'échec rapide pour les vraies erreurs."""
    if isinstance(exc, (TimeoutError, ConnectionError, http.client.RemoteDisconnected,
                        http.client.BadStatusLine, http.client.CannotSendRequest,
                        http.client.ResponseNotReady, ssl.SSLEOFError)):
        return True
    reason = getattr(exc, "reason", None)
    return isinstance(reason, (TimeoutError, ConnectionError))


_MAX_RETRY_AFTER = 15.0   # une lib interactive n'attend pas plus : au-delà, c'est l'hôte qui décide


# Un générateur PRIVÉ : tirer dans le `random` global décalait, à chaque relance, la séquence d'un hôte qui a posé
# `random.seed(...)` pour rendre ses tests reproductibles (compare.py et synthesis.py utilisent déjà les leurs).
_RNG = random.Random()


def _jitter() -> float:
    """Facteur dans [0,75 ; 1,0] — la formule du SDK officiel (`1 - 0,25 x aléa`).

    Sans lui, cent clients qui reçoivent la même erreur au même instant se
    réveillent TOUS à la même seconde et la resaturent : c'est ainsi qu'une
    panne d'une minute devient dix. Le délai demandé par le SERVEUR, lui, n'est
    jamais randomisé — il l'a calculé pour nous."""
    return 1.0 - 0.25 * _RNG.random()


def _retry_after(headers: dict[str, str]) -> float | None:
    """Le délai que le serveur demande, en secondes, ou None s'il n'en demande pas (ou si c'est illisible).

    Dans l'ordre du SDK officiel : `retry-after-ms` (millisecondes), puis `retry-after`
    en secondes, puis `retry-after` en DATE HTTP. Nul, négatif ou illisible : ignoré."""
    millisecondes = headers.get("retry-after-ms")
    if millisecondes:
        try:
            delai = float(millisecondes) / 1000.0
        except ValueError:
            delai = 0.0
        if delai > 0:
            return delai
    brut = headers.get("retry-after")
    if not brut:
        return None
    try:
        delai = float(brut)
    except ValueError:
        try:
            date = email.utils.parsedate_to_datetime(brut)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            delai = (date - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):    # OverflowError : une année géante (CPython <= 3.12)
            return None
    return delai if delai > 0 else None      # `nan > 0` est faux : ignoré aussi


def _retry_wait(exc: _HTTPStatusError, attempt: int) -> float:
    """Honour the server's Retry-After when present (capped), else backoff (with jitter)."""
    retry_after = _retry_after(exc.headers)
    if retry_after is not None:
        return min(retry_after, _MAX_RETRY_AFTER)
    return float(min(2**attempt, 8)) * _jitter()


def _send(url: str, body: bytes, headers: dict[str, str], timeout: float,
          stream: bool) -> tuple[http.client.HTTPResponse, str, str, int | None]:
    """Envoie la requête sur la connexion persistante de l'hôte ; rend la
    réponse (2xx) ou lève `_HTTPStatusError`. Les erreurs réseau remontent
    brutes (la connexion est jetée par l'appelant)."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ProviderError(f"Unsupported URL scheme in {url!r}")
    scheme, host, port = parts.scheme, parts.hostname or "", parts.port
    chemin = parts.path or "/"
    if parts.query:
        chemin += "?" + parts.query
    conn = _connection(scheme, host, port, timeout)
    conn.request("POST", chemin, body=body, headers=headers)
    response = conn.getresponse()
    if response.status >= 400:
        detail = response.read().decode("utf-8", errors="replace")
        raise _HTTPStatusError(response.status, detail,
                               {k.lower(): v for k, v in response.getheaders()})
    return response, scheme, host, port


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
    retries: int = _DEFAULT_RETRIES,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    entetes = {
        "content-type": "application/json",
        # urllib's default UA ("Python-urllib/x") is blocked by some
        # Cloudflare-fronted APIs (e.g. Groq → 403 error 1010). A normal
        # UA avoids that; callers can still override via `headers`.
        "user-agent": "autoagent/1.0",
        "content-length": str(len(body)),
        **(headers or {}),
    }
    _log.debug("POST %s (timeout=%s)", url, timeout)
    parts = urlsplit(url)
    for attempt in range(retries + 1):
        try:
            response, scheme, host, port = _send(url, body, entetes, timeout, stream=False)
            data = response.read().decode("utf-8")
        except UnicodeDecodeError as exc:
            # 0.24.0 : un corps qui n'est pas de l'UTF-8 (une page de passerelle, un octet corrompu) levait cette
            # exception BRUTE, que ni `ProviderError` ni le fail-over d'un hôte n'attrapent — trouvé par le banc de pannes.
            raise ProviderError(
                f"Provider returned a response body that is not valid UTF-8 ({exc.reason} at byte {exc.start})",
                retryable=False,
            ) from exc
        except _HTTPStatusError as exc:
            retryable = is_retryable_status(exc.code)
            if retryable and attempt < retries:  # 429/5xx: transient upstream — retry
                wait = _retry_wait(exc, attempt)
                _log.warning("HTTP %s from %s - retry %s/%s in %ss",
                             exc.code, url, attempt + 1, retries, wait)
                time.sleep(wait)
                continue
            _log.warning("HTTP %s from %s", exc.code, url)
            raise ProviderError(
                f"HTTP {exc.code} from {url}: {exc.detail}",
                status_code=exc.code, retryable=retryable,
            ) from exc
        except (OSError, http.client.HTTPException) as exc:
            # La connexion persistante est peut-être morte (fermée par le
            # serveur après inactivité) : on la jette, et on réessaie sur une
            # connexion neuve si l'erreur est transitoire.
            _discard(parts.scheme, parts.hostname or "", parts.port)
            if _is_transient(exc) and attempt < retries:
                wait = min(2 ** attempt, 8) * _jitter() if attempt else 0.0   # 1er réessai immédiat
                _log.warning("Transient network error for %s (%s) - retry %s/%s in %ss",
                             url, exc, attempt + 1, retries, wait)
                if wait:
                    time.sleep(wait)
                continue
            _log.warning("Request failed for %s: %s", url, exc)
            raise ProviderError(
                f"Request failed for {url}: {exc}", retryable=_is_transient(exc)
            ) from exc
        else:
            try:
                parsed: dict[str, Any] = json.loads(data)
                return parsed
            except json.JSONDecodeError as exc:
                raise ProviderError(f"Provider returned invalid JSON: {data[:500]}") from exc
    raise ProviderError(f"Request failed for {url}: retries exhausted")  # pragma: no cover


def post_sse(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
    retries: int = _DEFAULT_RETRIES,
    *,
    signals: dict[str, bool] | None = None,
) -> Iterator[dict[str, Any]]:
    """POST a JSON body and yield parsed ``data:`` events from an SSE stream.

    Server-Sent Events look like::

        event: content_block_delta
        data: {"type": "...", ...}

        data: {"another": "event"}

    We yield the JSON-decoded payload of every ``data:`` line. Lines that
    aren't ``data:`` (``event:``, comments, blanks) are skipped. The
    sentinel ``data: [DONE]`` (OpenAI-style) is swallowed (it is recorded in
    ``signals``) and reading goes on to the end of the response. Malformed
    ``data:`` payloads — not JSON, or JSON that is not an object — are
    skipped with a debug log rather than aborting the whole stream.

    Errors during the initial connection are retried with the same policy
    as ``post_json`` (429/5xx + transient network errors), then raise
    ``ProviderError``. A network failure MID-stream (connection reset, cut
    chunk, read timeout) raises a ``ProviderError`` too (0.24.0 — it used to
    propagate as the raw ``OSError`` / ``IncompleteRead``, which a host
    catching ``ProviderError`` never saw; the original exception is
    ``__cause__``, and ``retryable`` is True). A mid-stream retry would replay
    already-yielded events, so we never do it here.

    ``signals`` (0.24.0), when given, is filled with how the stream ENDED —
    ``{"done": <a ``[DONE]`` sentinel was seen>, "eof": <the server closed the
    response cleanly>}`` — so a provider can tell a stream that ended without
    its terminal marker (a proxy that closed: the text is TRUNCATED) from a
    complete one, without the events changing shape.

    The response is read to the end (or the connection discarded on error)
    so the persistent connection can be reused by the next call.
    """
    body = json.dumps(payload).encode("utf-8")
    entetes = {
        "content-type": "application/json",
        "accept": "text/event-stream",
        "user-agent": "autoagent/1.0",
        "content-length": str(len(body)),
        **(headers or {}),
    }
    if signals is not None:
        signals.update({"done": False, "eof": False})
    _log.debug("POST(SSE) %s (timeout=%s)", url, timeout)
    parts = urlsplit(url)
    response = None
    scheme, host, port = parts.scheme, parts.hostname or "", parts.port
    for attempt in range(retries + 1):
        try:
            response, scheme, host, port = _send(url, body, entetes, timeout, stream=True)
            break
        except _HTTPStatusError as exc:
            retryable = is_retryable_status(exc.code)
            if retryable and attempt < retries:
                wait = _retry_wait(exc, attempt)
                _log.warning("HTTP %s from %s (SSE) - retry %s/%s in %ss",
                             exc.code, url, attempt + 1, retries, wait)
                time.sleep(wait)
                continue
            _log.warning("HTTP %s from %s (SSE)", exc.code, url)
            raise ProviderError(
                f"HTTP {exc.code} from {url}: {exc.detail}",
                status_code=exc.code, retryable=retryable,
            ) from exc
        except (OSError, http.client.HTTPException) as exc:
            _discard(scheme, host, port)
            if _is_transient(exc) and attempt < retries:
                wait = min(2**attempt, 8) * _jitter() if attempt else 0.0
                _log.warning("Transient SSE error for %s (%s) - retry %s/%s in %ss",
                             url, exc, attempt + 1, retries, wait)
                if wait:
                    time.sleep(wait)
                continue
            _log.warning("SSE request failed for %s: %s", url, exc)
            raise ProviderError(
                f"Request failed for {url}: {exc}", retryable=_is_transient(exc)
            ) from exc
    assert response is not None  # loop either broke with a response or raised

    try:
        while True:
            raw_line = response.readline()
            if not raw_line:
                if signals is not None:
                    signals["eof"] = True
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                if signals is not None:
                    signals["done"] = True
                continue
            if not data:
                continue
            try:
                evenement = json.loads(data)
            except json.JSONDecodeError:
                _log.debug("Skipping non-JSON SSE data line: %.120s", data)
                continue
            if not isinstance(evenement, dict):
                # `null`, `[]`, `"x"`, `42` : pas un événement de fournisseur. Rendu tel quel, il faisait lever un
                # `AttributeError` brut dans le fournisseur (`event.get(...)`) — jamais une exception brute (0.24.0).
                _log.debug("Skipping non-object SSE data: %.120s", data)
                continue
            yield evenement
    except (OSError, http.client.HTTPException) as exc:
        # Une panne RÉSEAU en plein flux : la connexion n'est plus sûre, et l'hôte attend une `ProviderError`
        # (contrat de `ProviderError` : « network-level failures — DNS, timeout, connection reset »).
        _discard(scheme, host, port)
        _log.warning("SSE stream interrupted for %s: %s", url, exc)
        raise ProviderError(
            f"Stream interrupted for {url}: {type(exc).__name__}: {exc}", retryable=True,
        ) from exc
    except BaseException:
        # L'appelant a arrêté d'itérer (GeneratorExit), ou autre chose : la connexion n'est plus dans un état
        # sûr pour être réutilisée.
        _discard(scheme, host, port)
        raise
    finally:
        try:
            response.close()
        except Exception:
            pass
