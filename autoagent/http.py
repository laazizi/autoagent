from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from .errors import ProviderError
from .logging import get_logger

__all__ = ["post_json", "post_sse"]

_log = get_logger("http")

_DEFAULT_RETRIES = 2  # total attempts = retries + 1

# HTTP statuses worth retrying: rate limits and transient upstream failures.
# Other 4xx are caller errors — retrying them only wastes time and quota.
_RETRYABLE_HTTP = {429, 500, 502, 503, 504}


def _is_transient(exc: BaseException) -> bool:
    """Connection/handshake/read timeouts and resets — worth retrying. A DNS
    failure or a plain URLError (bad host) is NOT transient, so we don't retry
    it (keeps fast-fail behaviour for genuine errors)."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    reason = getattr(exc, "reason", None)
    return isinstance(reason, (TimeoutError, ConnectionError))


def _retry_wait(exc: urllib.error.HTTPError, attempt: int) -> float:
    """Honour the server's Retry-After when present (capped), else backoff."""
    try:
        retry_after = float((exc.headers or {}).get("Retry-After", ""))
        if retry_after > 0:
            return min(retry_after, 15.0)
    except (TypeError, ValueError):
        pass
    return float(min(2**attempt, 8))


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
    retries: int = _DEFAULT_RETRIES,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "content-type": "application/json",
            # urllib's default UA ("Python-urllib/x") is blocked by some
            # Cloudflare-fronted APIs (e.g. Groq → 403 error 1010). A normal
            # UA avoids that; callers can still override via `headers`.
            "user-agent": "autoagent/1.0",
            **(headers or {}),
        },
        method="POST",
    )
    _log.debug("POST %s (timeout=%s)", url, timeout)
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in _RETRYABLE_HTTP
            if retryable and attempt < retries:  # 429/5xx: transient upstream — retry
                wait = _retry_wait(exc, attempt)
                _log.warning("HTTP %s from %s - retry %s/%s in %ss",
                             exc.code, url, attempt + 1, retries, wait)
                time.sleep(wait)
                continue
            _log.warning("HTTP %s from %s", exc.code, url)
            raise ProviderError(
                f"HTTP {exc.code} from {url}: {detail}",
                status_code=exc.code, retryable=retryable,
            ) from exc
        except OSError as exc:  # URLError, TimeoutError, ConnectionError, ssl errors…
            if _is_transient(exc) and attempt < retries:
                wait = min(2 ** attempt, 8)
                _log.warning("Transient network error for %s (%s) - retry %s/%s in %ss",
                             url, exc, attempt + 1, retries, wait)
                time.sleep(wait)
                continue
            _log.warning("Request failed for %s: %s", url, exc)
            raise ProviderError(
                f"Request failed for {url}: {exc}", retryable=_is_transient(exc)
            ) from exc
        else:
            try:
                return json.loads(data)
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
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "content-type": "application/json",
            "accept": "text/event-stream",
            "user-agent": "autoagent/1.0",
            **(headers or {}),
        },
        method="POST",
    )
    _log.debug("POST(SSE) %s (timeout=%s)", url, timeout)
    response = None
    for attempt in range(retries + 1):
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in _RETRYABLE_HTTP
            if retryable and attempt < retries:
                wait = _retry_wait(exc, attempt)
                _log.warning("HTTP %s from %s (SSE) - retry %s/%s in %ss",
                             exc.code, url, attempt + 1, retries, wait)
                time.sleep(wait)
                continue
            _log.warning("HTTP %s from %s (SSE)", exc.code, url)
            raise ProviderError(
                f"HTTP {exc.code} from {url}: {detail}",
                status_code=exc.code, retryable=retryable,
            ) from exc
        except urllib.error.URLError as exc:
            if _is_transient(exc) and attempt < retries:
                wait = min(2**attempt, 8)
                _log.warning("Transient SSE error for %s (%s) - retry %s/%s in %ss",
                             url, exc, attempt + 1, retries, wait)
                time.sleep(wait)
                continue
            _log.warning("SSE request failed for %s: %s", url, exc)
            raise ProviderError(
                f"Request failed for {url}: {exc}", retryable=_is_transient(exc)
            ) from exc
    assert response is not None  # loop either broke with a response or raised

    with response:
        for raw_line in response:
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
