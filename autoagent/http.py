from __future__ import annotations

import http.client
import json
import ssl
import threading
import time
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlsplit

from .errors import ProviderError
from .logging import get_logger

__all__ = ["post_json", "post_sse", "close_connections"]

_log = get_logger("http")

_DEFAULT_RETRIES = 2  # total attempts = retries + 1

# HTTP statuses worth retrying: rate limits and transient upstream failures.
# Other 4xx are caller errors — retrying them only wastes time and quota.
_RETRYABLE_HTTP = {429, 500, 502, 503, 504}

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


def _retry_wait(exc: _HTTPStatusError, attempt: int) -> float:
    """Honour the server's Retry-After when present (capped), else backoff."""
    try:
        retry_after = float(exc.headers.get("retry-after", ""))
        if retry_after > 0:
            return min(retry_after, 15.0)
    except (TypeError, ValueError):
        pass
    return float(min(2**attempt, 8))


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
        except _HTTPStatusError as exc:
            retryable = exc.code in _RETRYABLE_HTTP
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
                wait = min(2 ** attempt, 8) if attempt else 0.0   # 1er réessai immédiat
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
) -> Iterator[dict[str, Any]]:
    """POST a JSON body and yield parsed ``data:`` events from an SSE stream.

    Server-Sent Events look like::

        event: content_block_delta
        data: {"type": "...", ...}

        data: {"another": "event"}

    We yield the JSON-decoded payload of every ``data:`` line. Lines that
    aren't ``data:`` (``event:``, comments, blanks) are skipped. The
    sentinel ``data: [DONE]`` (OpenAI-style) is swallowed — the iterator
    just ends. Malformed ``data:`` payloads are skipped with a debug log
    rather than aborting the whole stream.

    Errors during the initial connection are retried with the same policy
    as ``post_json`` (429/5xx + transient network errors), then raise
    ``ProviderError``. Errors mid-stream propagate as the underlying
    exception so the caller's ``try/except`` around iteration can decide
    what to do (the agent treats them as a failed turn) — a mid-stream
    retry would replay already-yielded events, so we never do it here.

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
    _log.debug("POST(SSE) %s (timeout=%s)", url, timeout)
    parts = urlsplit(url)
    response = None
    scheme, host, port = parts.scheme, parts.hostname or "", parts.port
    for attempt in range(retries + 1):
        try:
            response, scheme, host, port = _send(url, body, entetes, timeout, stream=True)
            break
        except _HTTPStatusError as exc:
            retryable = exc.code in _RETRYABLE_HTTP
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
                wait = min(2**attempt, 8) if attempt else 0.0
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
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if not data or data == "[DONE]":
                continue
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                _log.debug("Skipping non-JSON SSE data line: %.120s", data)
                continue
    except BaseException:
        # Flux interrompu (erreur réseau, ou l'appelant a arrêté d'itérer) :
        # la connexion n'est plus dans un état sûr pour être réutilisée.
        _discard(scheme, host, port)
        raise
    finally:
        try:
            response.close()
        except Exception:
            pass
